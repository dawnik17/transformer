class CustomAdam:
    def __init__(self, dimension, optimizer, warmup_steps=4000):
        """
        Attention Is All You Need

        Args:
            dimension (_type_): _description_
            optimizer (_type_): _description_
            warmup_steps (int, optional): _description_. Defaults to 4000.
        """
        self.optimizer = optimizer
        self.step_num = 0
        self.dimension = dimension
        self.warmup_steps = warmup_steps

    def step(self):
        self.step_num += 1
        lr = self.rate()

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.step()

    def rate(self):
        return self.dimension ** (-0.5) * min(
            self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5)
        )