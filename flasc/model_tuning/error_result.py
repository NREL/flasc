import numpy as np
import matplotlib.pyplot as plt


class ErrorResult():

    def __init__(self, params, errors, param_name="Parameter"):
        self.params = params
        self.errors = errors
        self.param_name = param_name

    def get_best_param(self):
        return self.params[np.argmin(self.errors)]
    
    def get_best_error(self):
        return np.min(self.errors)
    
    def plot_error(self, ax):
        ax.plot(self.params, self.errors)
        ax.set_xlabel(self.param_name)
        ax.set_ylabel("Error")
        ax.grid(True)

# Check if name is main
if __name__ == "__main__":
    
    er_res = ErrorResult([1,2,3], [7,5,6], "Test")

    print(er_res.get_best_param())
    print(er_res.get_best_error())

    fig, ax = plt.subplots()
    er_res.plot_error(ax)
    plt.show()
