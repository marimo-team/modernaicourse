import torch
from torchvision.datasets import MNIST
from torch.utils._python_dispatch import TorchDispatchMode
from collections import Counter
import mugrade
import pytest
import dis


images = MNIST(".", train=True, download=True)
idx = torch.where((images.targets==0) | (images.targets==1))[0]
images.data = images.data[idx]/255
images.targets = images.targets[idx]


multiply_ops = [
    torch.ops.aten.matmul.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.addbmm.default,
    torch.ops.aten.baddbmm.default,
    torch.ops.aten.linear.default,
    torch.ops.aten.mv.default,
    torch.ops.aten.addmv.default,
    torch.ops.aten.dot.default,
    torch.ops.aten.vdot.default,
    torch.ops.aten.inner.default,
    torch.ops.aten.tensordot.default,
    torch.ops.aten.linalg_vecdot.default,
]

class PreventTorchOps(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
            x, y = args[0], args[1]
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                if x.ndim > 0 and y.ndim > 0:
                    raise AssertionError("Vectorized addition not allowed")
                
        if func in multiply_ops:
            raise AssertionError("PyTorch matrix/vector products not allowed")

        kwargs = kwargs or {}
        return func(*args, **kwargs)


class MugradeSubmitAssertion:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is AssertionError:
            mugrade.submit(True)
            return True   # suppress the AssertionError
        if exc_type is None:
            mugrade.submit(False)
            return True   # suppress returning normally
        mugrade.submit(False)
        return False
    
def check_function_calls(f, func_name):
    inst = [a for a in dis.get_instructions(f) if a.argval == func_name]
    return len(inst) > 0
    

def test_classify_zero_one(classify_zero_one):
    # test on the first two images
    assert(classify_zero_one(images.data[0]) == 0)
    assert(classify_zero_one(images.data[1]) == 1)

    # test on the first 100 items, make sure accuracy >90%
    n = 100
    preds = [(classify_zero_one(images.data[i]), images.targets[i].item()) 
             for i in range(n)]
    counts = Counter(preds)
    accuracy = (counts[(0,0)] + counts[(1,1)])/n
    assert(accuracy > 0.9)

def submit_classify_zero_one(classify_zero_one):
    # submissions for 20 items
    n = 20
    for i in range(n):
        mugrade.submit(classify_zero_one(images.data[i+100]))


def test_vector_add(vector_add):
    a,b = torch.randn(5), torch.randn(5)
    with PreventTorchOps():
        z = vector_add(a,b)
    assert(torch.allclose(z,a+b))
    
    with pytest.raises(AssertionError):
        vector_add(torch.randn(5), torch.randn(6))


def submit_vector_add(vector_add):
    a,b = torch.linspace(0,1,7), torch.linspace(2,3,7)
    with PreventTorchOps():
        z = vector_add(a,b)
    mugrade.submit(z.numpy())
    mugrade.submit(type(z))

    with MugradeSubmitAssertion():
        vector_add(a, torch.randn(5))


def test_vector_inner_product(vector_inner_product):
    a,b = torch.randn(5), torch.randn(5)
    with PreventTorchOps():
        z = vector_inner_product(a,b)
    assert(torch.allclose(torch.tensor(z),a@b))
    assert(isinstance(z, float))
    
    with pytest.raises(AssertionError):
        vector_inner_product(torch.randn(5), torch.randn(6))


def submit_vector_inner_product(vector_inner_product):
    a,b = torch.linspace(0,1,7), torch.linspace(2,3,7)
    with PreventTorchOps():
        z = vector_inner_product(a,b)
    mugrade.submit(z)
    mugrade.submit(type(z))

    with MugradeSubmitAssertion():
        vector_inner_product(a, torch.randn(5))


def test_matrix_vector_product_1(matrix_vector_product_1):
    a,b = torch.randn(5,4), torch.randn(4)
    with PreventTorchOps():
        z = matrix_vector_product_1(a,b)
    assert(torch.allclose(z,a@b))
    assert(check_function_calls(matrix_vector_product_1, "vector_inner_product"))
    assert(not check_function_calls(matrix_vector_product_1, "vector_add"))
    
    with pytest.raises(AssertionError):
        matrix_vector_product_1(a, torch.randn(6))


def submit_matrix_vector_product_1(matrix_vector_product_1):
    a,b = torch.linspace(0,1,8*7).reshape(8,7), torch.linspace(2,3,7)
    with PreventTorchOps():
        z = matrix_vector_product_1(a,b)
    mugrade.submit(z.numpy())
    mugrade.submit(type(z))
    mugrade.submit(check_function_calls(matrix_vector_product_1, "vector_inner_product"))
    with MugradeSubmitAssertion():
        matrix_vector_product_1(a, torch.randn(5))


def test_matrix_vector_product_2(matrix_vector_product_2):
    a,b = torch.randn(5,4), torch.randn(4)
    with PreventTorchOps():
        z = matrix_vector_product_2(a,b)
    assert(torch.allclose(z,a@b))
    assert(not check_function_calls(matrix_vector_product_2, "vector_inner_product"))
    assert(check_function_calls(matrix_vector_product_2, "vector_add"))
    
    with pytest.raises(AssertionError):
        matrix_vector_product_2(a, torch.randn(6))


def submit_matrix_vector_product_2(matrix_vector_product_2):
    a,b = torch.linspace(0,1,8*7).reshape(8,7), torch.linspace(2,3,7)
    with PreventTorchOps():
        z = matrix_vector_product_2(a,b)
    mugrade.submit(z.numpy())
    mugrade.submit(type(z))
    mugrade.submit(check_function_calls(matrix_vector_product_2, "vector_add"))
    with MugradeSubmitAssertion():
        matrix_vector_product_2(a, torch.randn(5))



def test_vector_matrix_product_2(vector_matrix_product_2):
    a,b = torch.randn(4,5), torch.randn(4)
    with PreventTorchOps():
        z = vector_matrix_product_2(b,a)
    assert(torch.allclose(z,b@a))
    assert(not check_function_calls(vector_matrix_product_2, "vector_inner_product"))
    assert(check_function_calls(vector_matrix_product_2, "vector_add"))
    
    with pytest.raises(AssertionError):
        vector_matrix_product_2(torch.randn(6), a)


def submit_vector_matrix_product_2(vector_matrix_product_2):
    a,b = torch.linspace(0,1,8*7).reshape(7,8), torch.linspace(2,3,7)
    with PreventTorchOps():
        z = vector_matrix_product_2(b,a)
    mugrade.submit(z.numpy())
    mugrade.submit(type(z))
    mugrade.submit(check_function_calls(vector_matrix_product_2, "vector_add"))
    with MugradeSubmitAssertion():
        vector_matrix_product_2(torch.randn(5), a)


def test_matmul_1(matmul_1):
    a,b = torch.randn(4,5), torch.randn(5,6)
    with PreventTorchOps():
        z = matmul_1(a,b)
    assert(torch.allclose(z,a@b))
    assert(check_function_calls(matmul_1, "vector_inner_product"))
    assert(not check_function_calls(matmul_1, "vector_add"))
    assert(not check_function_calls(matmul_1, "vector_matrix_product_2"))
    assert(not check_function_calls(matmul_1, "matrix_vector_product_1"))
    assert(not check_function_calls(matmul_1, "matrix_vector_product_2"))

    with pytest.raises(AssertionError):
        matmul_1(a, torch.randn(4,5))


def submit_matmul_1(matmul_1):
    a,b = torch.linspace(0,1,3*8).reshape(3,8), torch.linspace(2,3,8*5).reshape(8,5)
    with PreventTorchOps():
        z = matmul_1(a,b)
    mugrade.submit(z.numpy())
    mugrade.submit(type(z))
    mugrade.submit(check_function_calls(matmul_1, "vector_inner_product"))
    with MugradeSubmitAssertion():
        matmul_1(a, torch.randn(4,5))


def test_matmul_2(matmul_2):
    a,b = torch.randn(4,5), torch.randn(5,6)
    with PreventTorchOps():
        z = matmul_2(a,b)
    assert(torch.allclose(z,a@b))
    assert(not check_function_calls(matmul_2, "vector_inner_product"))
    assert(not check_function_calls(matmul_2, "vector_add"))
    assert(not check_function_calls(matmul_2, "vector_matrix_product_2"))
    assert(check_function_calls(matmul_2, "matrix_vector_product_1") or
           check_function_calls(matmul_2, "matrix_vector_product_2"))

    with pytest.raises(AssertionError):
        matmul_2(a, torch.randn(4,5))


def submit_matmul_2(matmul_2):
    a,b = torch.linspace(0,1,3*8).reshape(3,8), torch.linspace(2,3,8*5).reshape(8,5)
    with PreventTorchOps():
        z = matmul_2(a,b)
    mugrade.submit(z.numpy())
    mugrade.submit(type(z))
    mugrade.submit(check_function_calls(matmul_2, "matrix_vector_product_1") or
                   check_function_calls(matmul_2, "matrix_vector_product_2"))
    with MugradeSubmitAssertion():
        matmul_2(a, torch.randn(4,5))
        
        
def test_matmul_3(matmul_3):
    a,b = torch.randn(4,5), torch.randn(5,6)
    with PreventTorchOps():
        z = matmul_3(a,b)
    assert(torch.allclose(z,a@b))
    assert(not check_function_calls(matmul_3, "vector_inner_product"))
    assert(not check_function_calls(matmul_3, "vector_add"))
    assert(check_function_calls(matmul_3, "vector_matrix_product_2"))
    assert(not check_function_calls(matmul_3, "matrix_vector_product_1"))
    assert(not check_function_calls(matmul_3, "matrix_vector_product_2"))

    with pytest.raises(AssertionError):
        matmul_3(a, torch.randn(4,5))


def submit_matmul_3(matmul_3):
    a,b = torch.linspace(0,1,3*8).reshape(3,8), torch.linspace(2,3,8*5).reshape(8,5)
    with PreventTorchOps():
        z = matmul_3(a,b)
    mugrade.submit(z.numpy())
    mugrade.submit(type(z))
    mugrade.submit(check_function_calls(matmul_3, "vector_matrix_product_2"))
    with MugradeSubmitAssertion():
        matmul_3(a, torch.randn(4,5))



def test_batch_matmul(batch_matmul):
    a,b = torch.randn(2,3,4,5), torch.randn(2,3,5,6)
    with PreventTorchOps():
        z = batch_matmul(a,b)
    assert(torch.allclose(z,a@b))
    assert(check_function_calls(batch_matmul, "matmul_1") or
           check_function_calls(batch_matmul, "matmul_2") or
           check_function_calls(batch_matmul, "matmul_3"))
    assert(not check_function_calls(batch_matmul, "vector_add"))
    assert(not check_function_calls(batch_matmul, "vector_matrix_product_2"))
    assert(not check_function_calls(batch_matmul, "matrix_vector_product_1"))
    assert(not check_function_calls(batch_matmul, "matrix_vector_product_2"))

    with pytest.raises(AssertionError):
        batch_matmul(a, torch.randn(2,3,4,5))
    with pytest.raises(AssertionError):
        batch_matmul(a, torch.randn(4,3,5,6))
    with pytest.raises(AssertionError):
        batch_matmul(a, torch.randn(3,5,6))


def submit_batch_matmul(batch_matmul):
    a,b = torch.linspace(0,1,4*2*3*8).reshape(4,2,3,8), torch.linspace(2,3,4*2*8*5).reshape(4,2,8,5)
    with PreventTorchOps():
        z = batch_matmul(a,b)
    mugrade.submit(z.numpy())
    mugrade.submit(type(z))
    mugrade.submit(check_function_calls(batch_matmul, "matmul_1") or
                   check_function_calls(batch_matmul, "matmul_2") or
                   check_function_calls(batch_matmul, "matmul_3"))
    with MugradeSubmitAssertion():
        batch_matmul(a, torch.randn(4,2,3,8))
    with MugradeSubmitAssertion():
        batch_matmul(a, torch.randn(2,2,3,8))
    with MugradeSubmitAssertion():
        batch_matmul(a, torch.randn(2,8,5))

        

def test_block_matmul(block_matmul):
    a,b = torch.randn(16,12), torch.randn(12,8)
    with PreventTorchOps():
        z = block_matmul(a,b)
    assert(torch.allclose(z,a@b))

    assert(not check_function_calls(block_matmul, "matmul_1") and
           not check_function_calls(block_matmul, "matmul_2") and
           not check_function_calls(block_matmul, "matmul_3"))

    with pytest.raises(AssertionError):
        block_matmul(a, torch.randn(16,12))
    with pytest.raises(AssertionError):
        block_matmul(a, torch.randn(12,7))


def submit_block_matmul(block_matmul):
    a,b = torch.linspace(0,1,8*16).reshape(8,16), torch.linspace(2,3,16*4).reshape(16,4)
    with PreventTorchOps():
        z = block_matmul(a,b)
    mugrade.submit(z.numpy())
    mugrade.submit(type(z))
    mugrade.submit(check_function_calls(block_matmul, "matmul_1") or
                   check_function_calls(block_matmul, "matmul_2") or
                   check_function_calls(block_matmul, "matmul_3"))
    with MugradeSubmitAssertion():
        block_matmul(a, torch.randn(8,4))
    with MugradeSubmitAssertion():
        block_matmul(a, torch.randn(16,6))