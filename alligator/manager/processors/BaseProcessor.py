class BaseProcessor:
    # registry containing all processor classes
    registry = {}
    # current processor ID
    processor_id = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.processor_id:
            BaseProcessor.registry[cls.processor_id] = cls

    def process(self, data):
        raise NotImplementedError("Processor must be implemented")
