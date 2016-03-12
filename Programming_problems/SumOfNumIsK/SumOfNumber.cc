#include "SumOfNumber.hpp"

//计算n个数中选择k个数其和是否为m,选择数至少大于0

bool dfs_sum(int step,const int* numbers, int n, int expect_sum, int sum, int selected_num)
{
	if (sum == expect_sum && selected_num != 0)
	{
		return true;
	}

	if (step == n)
	{
		return false;
	}
	else
	{
		return dfs_sum(step + 1, numbers, n, expect_sum, sum + numbers[step], selected_num+1)
				||  dfs_sum(step + 1, numbers, n, expect_sum, sum, selected_num);
	}
}

bool judge_sum_is_k(const int* numbers,  int n, int sum)
{
	return dfs_sum(0, numbers,  n, sum, 0, 0);
}
