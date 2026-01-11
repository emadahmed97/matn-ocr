from metaflow import FlowSpec, step


class Artifacts(FlowSpec):
    """A flow that showcases how artifacts work."""


    """
    assignment 2a):
    Create a simple flow that tracks a sequence of numerical operations. 
    In the first step, initialize an artifact with a number. 
    In each subsequent step, update the artifact by applying a different arithmetic operation 
    (e.g., addition, subtraction, multiplication) and append each new value to a list. 
    In the final step, print the entire history 
    of values and calculate both the sum and average.

    """
    @step
    def start(self):
        """Initialize the variable."""
        self.num = 2
        print("Initial value:", self.num)
        self.list_of_values = [self.num]
        self.next(self.multiply)

    @step
    def multiply(self):
        """Multiply the value of the variable."""
        self.num = self.num * 3
        self.list_of_values.append(self.num)
        self.next(self.subtraction)

    @step
    def subtraction(self):
        """Subtract from the value of the variable."""
        self.num = self.num - 4
        self.list_of_values.append(self.num)
        self.next(self.addition)

    @step
    def addition(self):
        """Add to the value of the variable."""
        self.num = self.num + 5
        self.list_of_values.append(self.num)
        self.next(self.increment)

    @step
    def increment(self):
        """Increment the value of the variable."""
        print("Incrementing the variable by 2")
        self.num += 2
        self.list_of_values.append(self.num)
        self.next(self.end)

    @step
    def end(self):
        """Print the final value of the variable."""
        print("Final list:", self.list_of_values)
        sum_of_values = sum(self.list_of_values)
        avg_of_values = sum_of_values / len(self.list_of_values)
        print("Sum of values:", sum_of_values)
        print("Average of values:", avg_of_values)


if __name__ == "__main__":
    Artifacts()
