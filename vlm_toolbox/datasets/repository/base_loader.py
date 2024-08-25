import asyncio
from typing import Any


class BaseLoader:
    @classmethod
    def load(cls, uri: str, **kwargs: Any) -> Any:
        raise NotImplementedError
    
    @classmethod
    async def load_async(cls, uri: str, *args, **kwargs: Any) -> Any:
        """
        Asynchronously load data.

        This method is similar to `load`, but it is designed to run asynchronously, 
        which can be beneficial for loading large datasets or when integrating with 
        asynchronous workflows.

        Args:
            uri (str): The identifier of the data to load.
            *args (Any): Additional positional arguments passed to the specific loading method.
            **kwargs (Any): Additional keyword arguments passed to the specific 
                loading method.

        Returns:
            Any: The loaded data, whose type depends on the file format being loaded.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, cls.load, uri, *args, **kwargs)

    
__all__ = ["BaseLoader"]
