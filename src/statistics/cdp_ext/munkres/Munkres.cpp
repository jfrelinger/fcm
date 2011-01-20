/*
 * Munkres.cpp
 *
 *  Created on: Sep 29, 2010
 *      Author: jolly
 */

#include "Munkres.h"
#include "../matrix/newmat.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <limits>
#define WANT_STREAM
#include "../matrix/newmatio.h"

path_item::path_item(int i, int j, path_type p_or_s) {
	row = i;
	col = j;
	type = p_or_s;
}

path_item::~path_item() {

}

Munkres::Munkres() {
	// TODO Auto-generated constructor stub

}

Munkres::~Munkres() {
	// TODO Auto-generated destructor stub
}

Matrix Munkres::solve(SquareMatrix icost) {
	cost = icost;
	size = cost.Ncols();
	starred.ReSize(size);
	covered_rows.ReSize(size);
	covered_cols.ReSize(size);
	primed.ReSize(size);
	starred = 0;
	primed = 0;
	covered_rows = 0;
	covered_cols = 0;
	k = min_uncovered();

	step1();

	return starred;
}

void Munkres::step1() {
	/*
	 * subtract the smallest element from each row from that row.
	 * goto step 1
	 */
	for(int i=1; i<= size; i++)
	{
		RowVector a = cost.Row(i);
		double m = a.Minimum();
		a << a - m;
		cost.Row(i) << a;
	}
	step2();
}
void Munkres::step2() {
	/*
	 * find a zero, if now starred zeros in row or column star z
	 * repeat for each element.
	 *
	 * goto step 3
	 */
	for(int i=1; i<= size; i++)
	{
		for(int j=1; j<= size; j++)
		{

			if (cost(i,j) == 0)
			{
				if (!is_starred_in_row_col(i,j))
				{
					starred(i,j) = 1;
				}
			}
		}
	}
	step3();
}
void Munkres::step3() {
	/* cover each coulumn containing a starred zero
	 * if size covered columns we're done.
	 * else goto step 4
	 */
	int cov_count = 0;
	for(int i=1; i<=size; i++)
	{
		for(int j=1; j<=size; j++)
		{
			if (starred(i,j)==1)
			{
				cover_col(j);
				cov_count += 1;
			}
		}
	}
	if (cov_count != size)
	{
		step4();
	}

}
void Munkres::step4() {
	/* find an uncovered zero and prime it
	 * if now starred zeros in row goto step 5
	 * else cover row and uncover colum of starred zero
	 *
	 * once no uncovered exist goto step 6.
	 */
	bool done = false;
	while(!done){
		int i,j;
		if (find_zero(cost,&i,&j))
		{
			if (!is_covered(i,j))
			{
				prime(i,j);
				int a = starred_in_row(i);
				if (a==0) // if no starred zeros
				{
					done = true;
					step5(i,j);
				}
				else
				{
					uncover_col(a);
					cover_row(i);
				}

			}
		}
		else
		{
			done = true;
			step6(min_uncovered());
		}
	}
}
void Munkres::step5(int i, int j) {
	/* take a primed zero, and construct a list of...
	 * 1. a starred zero in it's column (if it exists)
	 * 2. if there's a starred zero there will be a primed zero in its row.
	 *
	 * then
	 * unstar the starred,
	 * star all the primes
	 * erase all primes
	 * uncover everything
	 * return to step 3.
	 */
	vector<path_item> path;
	path.push_back(path_item(i,j,PRIMED));
	bool done = false;
	int row = 0;
	int col = j;
	while(!done)
	{
		row = find_starred_zero_in_col(col);
		if (row)
		{
			path.push_back(path_item(row,col,STARRED));
			col = find_primed_zero_in_row(row);
			path.push_back(path_item(row,col,PRIMED));
		}
		else
		{
			done = true;
		}
	}

	for(int i=0; i < path.size(); i++)
	{
		path_item item = path[i];
		if (item.type==PRIMED) // primed so we star
		{
			starred(item.row, item.col) = 1;
		}
		else
		{ // we're starred so we unstar
			starred(item.row, item.col) = 0;
		}
	}
	primed = 0; // remove all primes
	covered_rows = 0; // uncover all covered lines
	covered_cols = 0;
	step3();
}
void Munkres::step6(Real val) {
	/* take a value and add it to ever covered row
	 * then subtract it from every uncovered column.
	 * return to step 4
	 */
	for (int i=1; i <= size; i++)
	{
		// fix this....
		if (!is_covered_col(i))
		{ // uncovered column
			cost.column(i) -= val;
		}
		if (is_covered_row(i))
		{
			cost.row(i) += val;
		}
	}
	step4();
}

bool Munkres::is_starred_in_row_col(int row, int col) {
	RowVector a = starred.row(row);
	ColumnVector b = starred.column(col);
	if (!a.IsZero()) {
		return true;
	}
	else if (!b.IsZero()) {
		return true;
	}
	else
	{
	return false;
	}
}

int Munkres::starred_in_row(int row) {
	// find a starred value in a row
	RowVector a = starred.row(row);
	for(int i=1;i<=size;i++)
	{
		if(a(i)==1)
		{
			return i;
		}
	}
	return 0;
}

void Munkres::cover_col(int col) {
	//cover a column
	covered_cols(col) = 1;
}

void Munkres::uncover_col(int col) {
	//uncover a column
	covered_cols(col) = 0;
}

void Munkres::cover_row(int row) {
	// cover a row
	covered_rows(row) = 1;
}

void Munkres::uncover_row(int row) {
	// uncover a row
	covered_rows(row) = 0;
}

bool Munkres::is_covered(int row, int col) {
	// check if a position in a covered row or column
	if ((covered_rows(row) == 1) || (covered_cols(col)==1))
	{
		return true;
	}
	else
	{
		return false;
	}
}

bool Munkres::is_covered_col(int col) {
	// check if a column is covered
	if (covered_cols(col) == 1) {
		return true;
	}
	else
	{
		return false;
	}
}

bool Munkres::is_covered_row(int row) {
	// check if a row is covered
	if (covered_rows(row)==1) {
		return true;
	}
	else
	{
		return false;
	}
}


void Munkres::prime(int row, int col) {
	// prime a postion
	primed(row,col) = 1;
}

bool Munkres::find_zero(SquareMatrix mat, int* row, int* col) {
	// find a zero thats uncovered
	for (int i =1; i<= size; i++)
	{
		for (int j = 1; j<= size; j++)
		{
			if (mat(i,j) == 0)
			{
				if (!is_covered(i,j))
				{
					*row = i;
					*col = j;
					return true;
				}
			}
		}
	}
	return false;
}

float Munkres::min_uncovered() {
	// find the minumum uncovered value in the cost matrix.

	Real min = std::numeric_limits<Real>::infinity();

	for (int i =1; i<= size; i++)
	{
		for (int j = 1; j<= size; j++)
		{
			if (!is_covered(i,j))
			{
				if(cost(i,j) < min)
				{
					min = cost(i,j);
				}
			}
		}
	}
	return min;
}

int Munkres::find_starred_zero_in_col(int col) {
	// given a column find a starred zero in it, otherwise return 0
	ColumnVector a = cost.column(col);
	for (int i = 1; i <= size; i++)
	{
		if ((starred(i,col) == 1) && (a(i) == 0)) // if it's starrred it should be zero but check just in case....
			return i;
	}
	return 0;
}

int Munkres::find_primed_zero_in_row(int row) {
	// given a row, find if it has a primed zero in it, otherwise return 0
	RowVector a = cost.row(row);
	for (int i = 1; i <= size; i++)
	{
		if ((primed(row,i) == 1) && (a(i) == 0)) // if it's primed it should be a zero, but check just in case...
			return i;
	}
	return 0;
}


//int main() {
//	std::cout << "MAIN" << std::endl;
//	Munkres m;
//	SquareMatrix cost(64);
	//cost << 1 << 2 << 3 << 2 <<4 << 6<< 3<<6<<9;
//	cost << 1<< 2 << 3 << 4
//		 << 2 << 4 << 6 << 8
//		 << 3 << 6 << 9 << 12
//		 << 16 << 12 << 8 << 4;
//	for (int i=1;i<=64;i++) {
//		for (int j=1;j<=64;j++) {
//			cost(i,j) = (64-i)*j+64-j;
//		}
//	}
//	std::cout << "starting" << std::endl;
//	std::cout << m.solve(cost);
//	return 0;
//}

