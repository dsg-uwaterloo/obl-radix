// Based on https://zenodo.org/records/15169905
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

int main() {
	uint64_t n = 1000000000;
	uint32_t* follower = new uint32_t[n + 1];
	memset(follower, 0, sizeof(uint32_t) * (n + 1));
	uint32_t* following = new uint32_t[n + 1];
	memset(following, 0, sizeof(uint32_t) * (n + 1));

	ifstream fin("links-anon.txt");
	string line;
	while (getline(fin, line), !line.empty()) {
		size_t pos;
		uint64_t x = stoll(line, &pos);
		uint64_t y = atoll(line.c_str() + pos + 1);
		if (x <= n) ++follower[x];
		if (y <= n) ++following[y];
	}
	fin.close();

	ofstream file1n2_1("twitter_1n2_1.txt"); // ID follows popular
	ofstream file1_2("twitter_1_2.txt"); // Inactive ID
	ofstream file2_1("twitter_2_1.txt");
	ofstream file2_2("twitter_2_2.txt"); // Normal ID
	ofstream file3_1("twitter_3_1.txt"); // ID follows normal
	ofstream file3_2("twitter_3_2.txt"); // Popular ID

	ifstream fin2("links-anon.txt");
	string line2;
	while (getline(fin2, line2), !line2.empty()) {
		size_t pos;
		uint64_t x = stoll(line2, &pos);
		uint64_t y = atoll(line2.c_str() + pos + 1);
		if (follower[x] >= 10000) {
			file1n2_1 << y << " X\n";
		};
		if ((0 < follower[x]) && (5 <= follower[x] || 5 <= following[x])) {
			file3_1 << y << " X\n";
		};
	}
	fin2.close();
	file1n2_1.close();
	file3_1.close();


	for (uint64_t i = 1; i <= n; ++i) {
		if (follower[i] > 0) {
			if (follower[i] >= 10000) {
				file3_2 << i << " X\n";
			}
			else if (follower[i] <= 5 && following[i] <= 5) {
				file1_2 << i << " X\n";
			}
			else {
				file2_2 << i << " X\n";
			}
		}
	}
	file1_2.close();
	file2_2.close();
	file3_2.close();
	
	delete[] follower;
	delete[] following;

	return 0;
}
