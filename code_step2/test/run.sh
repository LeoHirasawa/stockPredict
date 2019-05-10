#!/bin/bash

# 将参数存入一个数组并输出
echo "将参数存入一个数组并输出"
test_array=()
# @和*都可以表示“所有的”意思，@还可以将字符串形式的一串变量给拆分开。
for i in "$@"; do
    echo $i
    test_array=(${test_array[@]} $i)
done
echo ${test_array[@]}


# 变量创建，使用和修改
your_name="runoob.com"
echo ${your_name}
your_name="alibaba"
str="Hello, I know you are \"${your_name}\"! \n"
echo -e $str

myUrl="http://www.google.com"
readonly myUrl

# 循环执行语句
for file in $(ls ./); do
    # 双引号里面可以放变量，可以放转义字符。单引号则不可以
    echo "I am good at ${file} Script"
done

# 尝试执行python脚本（带参数）
python ./hybrid_model_script_mode.py --model_structure ${test_array[@]}