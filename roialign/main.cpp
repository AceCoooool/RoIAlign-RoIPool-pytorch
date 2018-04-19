#include <iostream>
#include <ATen/ATen.h>
#include "temp.h"

using namespace std;
using namespace at;


int main() {
    // different sample in this example has the same output, you can
    // test it on randn.
    auto feat = CPU(kFloat).arange(64).view({1, 1, 8, 8});
//    auto feat = CPU(kFloat).randn({1, 1, 8, 8}).view({1, 1, 8, 8});
    cout << feat << endl;
    float roi_data[] = {0, 1.6, 1.6, 9.2, 11.0};
    auto roi = CPU(kFloat).tensorFromBlob(roi_data, {1, 5});
    int64_t pool_h = 2, pool_w = 2, sample=1;
    double scale = 0.5;
    auto output = roi_align_forward_cpu(feat, roi, pool_h, pool_w, scale, sample);
    cout << output << endl;
//    auto output2 = roi_align_forward_cpu(feat, roi, pool_h, pool_w, scale, 2);
//    cout << output2 << endl;
}