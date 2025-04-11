from fastapi import Depends
from core.system import rag_system

async def get_rag_system():
    """의존성 주입을 위한 시스템 인스턴스 제공"""
    return rag_system
