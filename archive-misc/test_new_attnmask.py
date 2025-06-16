import torch


def attnmask_new(sz):
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

def attnmask_old(sz):
    return torch.log(torch.tril(torch.ones(sz, sz)))

def compare_masks(sz):
    new_mask = attnmask_new(sz)
    old_mask = attnmask_old(sz)
    
    # print("New Mask:")
    # print(new_mask)
    # print("\nOld Mask:")
    # print(old_mask)
    
    if torch.equal(new_mask, old_mask):
        #print("\nThe masks are equal.")
        return True
    else:
        print("\nThe masks are NOT equal.")
        raise ValueError("Masks differ, check implementation.")

if __name__ == "__main__":
    for i in range(1, 100):
        # print(f"Comparing masks of size {i}:")
        compare_masks(i)
        #print("\n" + "="*50 + "\n")
    print("All masks are equal for sizes 1 to 99.")

    # test time taken for size 256 of each implementation usint timeit
    import timeit
    size = 256
    new_time = timeit.timeit(lambda: attnmask_new(size), number=1000)
    old_time = timeit.timeit(lambda: attnmask_old(size), number=1000)
    print(f"New mask time for size {size}: {new_time:.6f} seconds")
    print(f"Old mask time for size {size}: {old_time:.6f} seconds")
    if new_time < old_time:
        print("New mask implementation is faster.")
    else:
        print("Old mask implementation is faster.")