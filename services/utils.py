from providers.factory import get_llm_provider


async def generate_conversation_title(first_message: str, llm=None) -> str:
    """根据第一条消息生成对话标题（5-10字）"""
    try:
        if llm is None:
            llm = get_llm_provider()
        
        title_messages = [
            {"role": "system", "content": "你是一个对话标题生成器。请根据用户的第一条消息生成一个5-10个字的简洁标题，不要加引号或任何修饰，直接返回标题本身。"},
            {"role": "user", "content": f"请为以下对话生成标题：{first_message}"}
        ]
        title = llm.chat(title_messages)
        # 清理标题：去除引号、换行等
        title = title.strip().strip('"').strip("'")
        # 限制长度
        if len(title) > 15:
            title = title[:15]
        return title if title else "新对话"
    except Exception:
        return "新对话"
