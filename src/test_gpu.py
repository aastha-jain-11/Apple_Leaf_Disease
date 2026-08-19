import torch


print("=" * 60)
print("PYTORCH / GPU TEST")
print("=" * 60)

print()

print("PyTorch version:")
print(torch.__version__)

print()

print("PyTorch CUDA version:")
print(torch.version.cuda)

print()

print("CUDA available:")
print(torch.cuda.is_available())

print()

if not torch.cuda.is_available():

    print("CUDA is NOT available.")
    raise SystemExit(1)


print("GPU count:")
print(torch.cuda.device_count())

print()

print("GPU name:")
print(torch.cuda.get_device_name(0))

print()

properties = torch.cuda.get_device_properties(0)

print("GPU memory:")
print(
    f"{properties.total_memory / (1024 ** 3):.2f} GB"
)

print()

print("Testing small GPU tensor...")

try:

    device = torch.device("cuda:0")

    x = torch.tensor(
        [1.0, 2.0, 3.0],
        device=device
    )

    y = torch.tensor(
        [4.0, 5.0, 6.0],
        device=device
    )

    z = x + y

    torch.cuda.synchronize()

    print("GPU tensor:")
    print(z)

    print()

    print("Tensor device:")
    print(z.device)

    print()

    print("=" * 60)
    print("GPU TEST PASSED")
    print("=" * 60)

except Exception as error:

    print()
    print("=" * 60)
    print("GPU TEST FAILED")
    print("=" * 60)

    print()
    print("Error:")
    print(error)

    raise