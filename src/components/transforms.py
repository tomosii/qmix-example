import torch as th


class Transform:
    def transform(self, tensor) -> th.Tensor:
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor: th.Tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        """

        """
        return (self.out_dim,), th.float32


if __name__ == "__main__":
    # サンプル
    onehot = OneHot(4)
    print(onehot.infer_output_info(None, None))
    print(onehot.transform(th.tensor([[1], [3]])))
