#include "SumOfNumber.hpp"
#include <gtest/gtest.h>

TEST(judge_sum_is_k, Numbers_sum)
{
	{
		int numbers[] = {1,2,3,4};
		 EXPECT_EQ(true, judge_sum_is_k(numbers, 4, 1));
		 EXPECT_EQ(true, judge_sum_is_k(numbers, 4, 4));
		 EXPECT_EQ(true, judge_sum_is_k(numbers, 4, 6));
		 EXPECT_EQ(true, judge_sum_is_k(numbers, 4, 10));
		 EXPECT_EQ(false, judge_sum_is_k(numbers, 4, 11));
		 EXPECT_EQ(false, judge_sum_is_k(numbers, 4, 0));
	}
	{
		int numbers[] = {1,1,2,2};
		 EXPECT_EQ(true, judge_sum_is_k(numbers, 4, 4));
		 EXPECT_EQ(true, judge_sum_is_k(numbers, 4, 5));
		 EXPECT_EQ(true, judge_sum_is_k(numbers, 4, 6));
		 EXPECT_EQ(false, judge_sum_is_k(numbers, 4, 7));
		 EXPECT_EQ(false, judge_sum_is_k(numbers, 4, 11));
		 EXPECT_EQ(false, judge_sum_is_k(numbers, 4, 0));
	}
	{
		int numbers[] = {1,10,100,1000};
		 EXPECT_EQ(true, judge_sum_is_k(numbers, 4, 1000));
		 EXPECT_EQ(true, judge_sum_is_k(numbers, 4, 100));
		 EXPECT_EQ(true, judge_sum_is_k(numbers, 4, 1111));
		 EXPECT_EQ(false, judge_sum_is_k(numbers, 4, 2));
		 EXPECT_EQ(false, judge_sum_is_k(numbers, 4, 2000));
		 EXPECT_EQ(false, judge_sum_is_k(numbers, 4, 500));
	}
	{
			int numbers[] = {1,-1};
			 EXPECT_EQ(true, judge_sum_is_k(numbers, 2, 0));
		}
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
