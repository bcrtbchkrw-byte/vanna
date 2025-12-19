import asyncio
from automation.data_maintenance import get_maintenance_manager
from core.logger import setup_logger

if __name__ == "__main__":
    setup_logger(level="INFO")
    manager = get_maintenance_manager()
    
    # FORCE RUN (Monkeypatch)
    manager.should_run_saturday_merge = lambda: True
    
    asyncio.run(manager.run_saturday_merge())
