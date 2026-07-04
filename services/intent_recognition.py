from typing import List, Dict, Optional, Any
from providers.factory import get_llm_provider
import json


class IntentRecognizer:
    """意图识别服务"""
    
    def __init__(self):
        self.llm = get_llm_provider()
    
    def recognize_intent(self, user_message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        识别用户意图

        意图类型与 services/agent.py 中 Agent 的工具使用原则对齐：
        - rag_query   → 对应 rag_chat_tool（本地知识库 / 民法典合同编）
        - web_search  → 对应 tavily_search_tool（实时网络信息）
        - chat        → 直接回答（常识 / 计算 / 推理 / 闲聊）
        - unknown     → 无法识别

        Args:
            user_message: 用户输入的消息
            context: 上下文信息（可选）

        Returns:
            意图识别结果，包含：
            - intent: 意图类型（rag_query, web_search, chat, unknown）
            - confidence: 置信度 (0-1)
            - entities: 提取的实体
            - reasoning: 识别理由
        """
        system_prompt = """
你是一个意图识别专家。请分析用户的输入，并识别其意图。

**意图类型定义：**
1. **rag_query**: 用户需要查询本地知识库内容。当问题涉及合同编法条、民事法律问题、文档内容、特定事实、概念、产品信息等，或以"什么是"、"如何"、"为什么"、"介绍"等开头时归为此类。本地知识库包含：中华人民共和国民法典合同编。
2. **web_search**: 用户需要获取实时网络信息。当问题涉及最新新闻、天气预报、股价汇率、实时事件、近期动态等需要联网才能获取的信息时归为此类。
3. **chat**: 用户进行普通聊天、闲聊、情感交流，或询问常识性问题、简单计算、逻辑推理等不需要外部信息即可直接回答的问题。
4. **unknown**: 无法识别的意图。

**工具使用原则（用于辅助判断）：**
- 当问题涉及合同编法条、民事法律问题、文档内容等知识库相关问题时，归为 rag_query
- 当需要获取最新信息、实时新闻、天气预报、股价汇率等网络实时信息时，归为 web_search
- 对于常识性问题、简单计算、逻辑推理等不需要外部信息的问题，归为 chat
- 只有在必要时才归为需要外部信息的意图（rag_query / web_search），避免过度分类

**要求：**
- 必须从上述四个意图类型中选择一个
- 输出格式必须是 JSON，包含以下字段：
  - "intent": 意图类型（rag_query/web_search/chat/unknown）
  - "confidence": 置信度，0-1之间的数字
  - "entities": 提取的关键实体（列表形式）
  - "reasoning": 识别理由（简洁说明为什么是这个意图）

**示例：**
用户输入："民法典合同编中关于违约责任是怎么规定的？"
输出：{"intent": "rag_query", "confidence": 0.95, "entities": ["民法典合同编", "违约责任"], "reasoning": "用户询问合同编法条，属于本地知识库相关内容"}

用户输入："今天A股大盘走势如何？"
输出：{"intent": "web_search", "confidence": 0.92, "entities": ["A股", "大盘走势"], "reasoning": "用户询问实时股价信息，需要联网搜索获取最新数据"}

用户输入："1+1等于几？"
输出：{"intent": "chat", "confidence": 0.95, "entities": [], "reasoning": "用户询问简单计算，可直接回答，无需外部信息"}

用户输入："你好，今天心情怎么样？"
输出：{"intent": "chat", "confidence": 0.9, "entities": [], "reasoning": "用户进行日常问候和闲聊"}
"""
        
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_message}
        ]
        
        response = self.llm.chat(messages)
        
        try:
            result = json.loads(response)
            # 验证必要字段
            if "intent" not in result or "confidence" not in result:
                return self._fallback_result(user_message)
            return result
        except (json.JSONDecodeError, ValueError):
            return self._fallback_result(user_message)
    
    def _fallback_result(self, user_message: str) -> Dict[str, Any]:
        """
        当解析失败时的默认结果

        通过简单规则判断意图（与 agent.py 的工具使用原则对齐）
        """
        lower_msg = user_message.lower()

        # 检查是否需要联网搜索（实时信息）
        web_keywords = ["今天", "最新", "最近", "实时", "现在", "新闻", "天气", "股价", "汇率", "行情", "热点"]
        if any(keyword in lower_msg for keyword in web_keywords):
            return {
                "intent": "web_search",
                "confidence": 0.7,
                "entities": [],
                "reasoning": "检测到实时信息关键词，需要联网搜索"
            }

        # 检查是否为知识库查询（合同编 / 法条 / 文档内容）
        rag_keywords = ["合同编", "民法典", "法条", "法律规定", "什么是", "什么", "如何", "为什么",
                        "介绍", "说明", "定义", "原理", "功能", "特点"]
        if any(keyword in lower_msg for keyword in rag_keywords):
            return {
                "intent": "rag_query",
                "confidence": 0.8,
                "entities": [],
                "reasoning": "检测到知识库查询关键词"
            }

        # 默认视为普通聊天（常识 / 计算 / 推理 / 闲聊）
        return {
            "intent": "chat",
            "confidence": 0.6,
            "entities": [],
            "reasoning": "默认归类为普通聊天"
        }


# 全局单例
_intent_recognizer = None

def get_intent_recognizer() -> IntentRecognizer:
    """获取意图识别器实例"""
    global _intent_recognizer
    if _intent_recognizer is None:
        _intent_recognizer = IntentRecognizer()
    return _intent_recognizer
