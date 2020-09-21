def findMaxSubArray(nums):
    cur_sum = nums[0]
    max_sum = nums[0]
    cur_array = [nums[0]]
    max_array = [nums[0]]
    for num in nums[1:]:
        if num > cur_sum + num:
            cur_sum = num
            cur_array = [num]
        else:
            cur_sum += num
            cur_array.append(num)
        if cur_sum > max_sum:
            max_sum = cur_sum
            max_array = cur_array.copy()
    return max_array


if __name__ == '__main__':
    nums = list(map(int, input().split()))
    res = findMaxSubArray(nums)
    print(res)

