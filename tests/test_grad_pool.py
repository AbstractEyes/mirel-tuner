import unittest

class GradPoolStepTest(unittest.TestCase):
    def test_step_updates_tensor_grad(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not available")
        from hooking.train.grad_pool import GradPool
        tensor = torch.zeros(1, requires_grad=True)
        pool = GradPool()
        pool.register(tensor, "t")
        pool.accumulate("t", torch.tensor([1.0]))
        pool.step()
        self.assertTrue(torch.allclose(tensor.grad, torch.tensor([1.0])))

if __name__ == "__main__":
    unittest.main()
