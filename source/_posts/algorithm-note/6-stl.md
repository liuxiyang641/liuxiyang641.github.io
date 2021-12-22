---
title: 6-stl
notshow: false
date: 2021-11-19 10:22:16
categories:
- algorithm
tags:
- book
---

# C++标准模板库

本文来自于《算法笔记》第六章内容

STL：standard template library

<!--more-->

## vector常见用法

可变长数组，使用前提

```c
#include <vector>
using namespace std;
```

定义方法：

```C
vector <typename> name;
// 比如：
vector <char> vec;
vector <vector <int> > twod_array // 注意这里的> >之间的空格，防止部分编译器编译错误
```

访问vector中的元素有两种方法，下标访问以及通过迭代器访问。

下标访问，和数组的正常访问一样，只要不越界即可：

```c
vector <int> vec;
// pass
a = vec[1];
```

迭代器访问，迭代器是一种类似于指针的东西，定义方法：

```c
vector <typename>::iterator it;
// 让it指向vector中的某个元素地址，然后直接使用*it就能取值，例如：
it = vec.begin();
val = *it;
val = *(it+3);
int i = 8;
val = *(it+i); // 注意这种迭代器+整数的写法，只有在vector和string中有实现，其它stl容器无提供相关内容
```

迭代器实现了自增和自减操作：

```c
// 可用于循环取vector中的元素
it++;
++it;
--it;
it--;
```

可以用于遍历元素：

```c
// 注意这里<vec.end()的写法，只有在vector和string容器里可用
for(vector<int>::iterator it = vec.begin(); it < vec.end(); ++it) {
  // pass
}
```

vector常用函数举例：

`begin()`和`end()`，需要注意的是`end()`不指向实际的元素，这是因为在c语言中，习惯用左闭右开的区间写法。

```c
// 返回vector的首地址以及尾地址的后一位
vec.begin();
vec.end();
// 这两个函数可以和iterator结合起来进行元素遍历，也可以用在sort函数中
sort(vec.begin(), vec.end());
```

`size()`返回元素个数，常用于遍历

```c
int total_num = vec.size();
for(int i=0; i < total_num; ++i) {
  // pass
}
```

 `push_back()`末尾增加元素

```c
for(int i = 0;i<n;++i) {
	scanf("%d", &temp);
	vec.push_back(temp);
}
```

`pop_back()`删除末尾元素

```c
vec.pop_back();
```

`clear()`清空所有元素

`insert(it, x)`在特定位置插入新元素

```c
vec.insert(vec.begin()+2, -1); // 在第3个元素插入-1
```

`earse()`删除特定元素或者一个区间的元素（左闭右开）

```c
// 删除特定迭代器位置的元素
vec.earse(vec.begin()+3);
// 删除一个区间的元素
vec.earse(vec.begin()+1, vec.begin()+3);
```

## set常见用法

在c语言中的`set`是一个**自动排序且无重复元素**的容器。

使用前提

```c
#include <set>
```

定义方法，与vector一样

```c
set<typename> name;
set<int> st;
```

另外，set的排序是可以更改的，默认的是使用`less<typename>`的排序标准，可以改为从大到小的排序

```c
set<int, greater<int>> numbers;
```

访问元素，只能使用迭代器，不能使用下标访问。遍历set中的元素：

```c
// 注意这里，不能使用it<st.end()来判断停止条件
for(set<int>::iterator it = st.begin(); it!=st.end(); ++it) {
  printf("%d ", *it);
}
```

`set`的常用函数：

`insert()`插入新元素，会去重并排序，记得没有`push_back()`。

```C
st.insert(1);
st.insert(2);
st.insert(3);
```

`size()`返回元素个数；

`find()`返回对应值的元素的迭代器：

```c
it = st.find(2);
// 如果没有找到对应元素
if (st.find(-1000) == st.end()) {
  printf("not found\n");
}
```

`earse()`删除元素：

```c
// 删除单个元素
st.earse(2); // 直接删除对应值的元素
st.earse(st.find(2)); // 删除对应迭代器位置的元素
// 删除一组元素，[)
st.earse(it1, it2);
```

`clear()`清空元素。

还有其它类似的`set`容器，比如`multiset`和`unordered_set`。

`multiset`不会去重，但是会排序，使用方法与set类似，同样是默认从小到大的排序。

`unordered_set`不会排序但是会去重，速度更快，可以用来很方便的去重元素，使用方法与set类似。

## string常见用法

c++在stl中提供了string类型用于实现对字符串的处理。

使用前提

```c
// 注意<string.h>是完全不同的头文件
#include<string>
```

定义：

```c
string str = "abcde";
```

输入与输出：

```c
// 只能使用cin和cout
cin>>str;
cout<<str;
// 输出也可以使用printf
printf("%s", str.c_str());
```

元素访问：

单元素访问，类似于vector可以使用下标：

```c
int = 2;
printf("%c", str[i]);
```

通过迭代器访问：

```c
string::iterator it = str.begin();
it++;
printf("%c", *it);
it+=3; // 类似于vector，可以让迭代器加一个常量长度
```

string常用函数：

操作符`+=`，直接拼接两个string，`str1+=str2`。

比较符`==,!=,<,<=,>,>=`，比较顺序是字典序。

元素个数查询`length()/size()`。

插入新字符`insert()`，举两个实例：

```c
// 在特定位置pos插入字符串str
str.insert(3, "opt");
// 在迭代器it，插入另一个字符串的[it1, it2)子串
str.insert(it, str2.begin(), str2.end());
```

删除元素`erase()`，举例：

```c
// 删除迭代器位置it的单个元素
str.erase(it);
// 删除一个区间的元素，两个迭代器中间的元素
str.erase(first, last);
// 删除一个位置开始的一定长度的元素
str.erase(pos, length);
```

清空元素`clear()`。

截取子串，`substr(pos, len)`。

寻找子串`find()`，举例：

```c
// 寻找一个子串，返回位置pos
int pos = str.find("ab");
// 如果没有找到，返回没有找到的代码
if (pos == string::npos) {
  printf("not found\n");
} // 记住，string::npos是一个常数，作为find函数匹配失败的返回值
// 还可以从某个位置开始匹配
pos = str.find("ab", 3);
```

替换子串`replace()`，举例：

```c
str.replace(2, 4, "gggg"); //把str中[2, 4)的子串替换为"gggg"
str.replace(it1, it2, "gggg"); //把str中[it1, it2)的子串替换为"gggg"
```

## map常见用法

map可以将任何基本类型（包括容器）映射到任何基本类型（包括容器）。

使用map

```c
#include <map>
```

map的定义

```c
map<keytype, valtype> mp;
// 如果使用字符串作为key，只能使用string
map<string, int> mp;
```

map元素的访问，通过下标key进行访问

```c
mp['aa'] = 2;
mp['aa'] = 20;
```

通过迭代器访问

```c
map<string, int>::iterator it;
it = mp.begin();
cout<<it->first; // 通过it->first访问键, it->second访问值
cout<<it->second;
```

map内部是使用红黑树实现的，因此会按照从小到大的顺序自动排列键。

map常用函数：

查找某个key是否存在`find()`，返回迭代器，`mp.find("aa")`；

插入元素`insert()`

```c
mp.insert(pair<string, int>("bbb", 7)); // 注意map都是以pair为操作对象，插入需要是一个pair对象
```

删除元素`erase()`，

```c
mp.erase(it); // 删除迭代器位置的元素
mp.erase("aa"); // 删除key键的元素
mp.erase(first_it, last_it); // 删除[first_it, last_it)的元素
```

元素个数`size()`；

清空`clear()`；

map中的键和值是唯一的，如果希望一个键对应多个值，可以使用`multimap`。

由于map会默认按照`less<typename>`进行排序，所以类似于set，c++11中提供了`unordered_map`。

map和set实际具有几乎完全相同的接口和函数名，set可以看做是一种特殊的map，即key=value。

## queue常见用法

queue是先进先出的限制性数据结构

使用queue

```c
#include<queue>
```

定义queue

```c
queue<typename> que;
```

访问元素，queue只能访问队首或者队尾，不能像前面的stl一样通过下标任意访问

```c
que.front(); // 访问队首，在使用前记住要先判断que.empty()，避免队空而出错
que.back(); // 访问队尾
```

queue常用函数

增加新元素`push()`。

队首元素出列`pop()`。

检测queue是否为空`empty()`，如果是空返回`true`。

元素个数`size()`。

## priority_queue常见用法

priority_queue是优先队列，和queue的区别是它保证队列中优先级最高的总是在*队首*，queue不会自动排序。

priority_queue的定义

```c
#include<queue>
priority_queue<typename> pq;
```

常用函数：

增加新元素`push()`。

查看队首元素`top()`，注意没有queue中的`front()`和`back()`。

弹出队首元素`pop()`，使用前记得使用`empty()`判断是否为空，防止报错。

检测空`empty()`。

元素个数`size()`。

如何设置priority_queue的优先级？

一般的，按照基本类型的从大到小排序，字符串按照字典序，int按照数值大小。

```c
priority_queue<int, vector<int>, less<int> > pq; // 优先级大的int元素在队首
```

其中的`vector<int>`是priority_queue的底层数据结构堆的容器，类型需要和前面的元素type保持一致。`less<int>`是数值大的在队首，这一点和前面的set是相反的，`greater<int>`表示数值小的会在队首。

另外，对于结构体，可以通过下面重载比较符`<`的方法定义优先级，注意只能重载`<`不能重载`>`，因为只要定义好了`<`，`>`和`==`也就都定义好了。

```c
struct fruit {
  string name;
  int price;
  friend bool operator < (fruit f1, fruit f2) {
    return f1.price < f2.price; // 让价格高的在队首，优先队列会选择f1和f2中比较出来大的对象放到前面
  }
};
priority_queue<fruit> fruits;
```

这种定义方式有些类似于`sort()`函数中可以自定义的比较函数，但是如果上面的比较方法放在`sort`中，会让价格低的在前面，与priority_queue刚好相反。

## stack常见用法

stack是后进先出的限制性容器，定义

```c
#include<stack>
stack<typename> st;
```

访问元素类似于priority_queue，只能通过`top()`访问栈顶元素。

常用函数：

`push()`增加新元素

`top()`获得栈顶元素

`pop()`退栈

`empty()`检测是否为空

`size()`元素个数

## pair常见用法

pair可以将两个元素合并为一个元素，可以看做是一个包含两个元素的struct。

pair的定义

```c
#include<utility>
// #include<map> // 由于map的内部使用了pair，所以map头文件中会自动添加utility，所以可以偷懒使用map头文件
pair<typename1, typename2> p;
```

pair的初始化

```c
pair<string, int> p ("aaa", 9);
```

pair临时构造

```c
pair<string, int>("cccc", 11);
// 或者使用make_pair函数
make_pair("cccc", 11);
```

pair元素访问，只有两个元素，分别是first和second

```c
p.first;
p.second;
```

pair常用函数

支持比较操作符，`==`,`<`,`>`等，规则是先比较first，只有first相等之后才会比较second。

## algorithm头文件下的常用函数

使用

```c
#include<algorithm>
```

`max`,`min`和`abs`分别返回最大值、最小值以及*整数*的绝对值

```c
int x = 1, y = -1, z = 3;
max(x, y);
min(x, y);
abs(y);
// 如果希望是浮点数的绝对值，使用<math>头文件下的fabs
fabs(-0.19);
```

`swap()`交换x和y的值

```c
swap(x, y);
```

`reverse()`反转一段数组或者一部分容器的元素

```c
int a [3] = {0, 1, 2};
reverse(a, a+2);
string b = "abcdefg";
reverse(b.begin(), b.begin()+3);
```

`next_permutation()`返回下一个排列

```c
do {
  printf("%d %d %d\n", a[0], a[1], a[2]);
} while (next_permutation(a, a+3));
```

`fill(it1, it2, val)`将数组或者容器的一部分连续元素赋值为相同值

```c
fill(a, a+2, 4);
```

`sort(it1, it2, compare_func)`排序数组或容器，结构体等。可能是最常用的方法。

```c
// 默认从小到大排序
sort(a, a+3);
// 从大到小排序数组
bool cmp(int x, int y) {
  return x > y;
}
sort(a, a+3, cmp);
// 排序容器，只有vector，string，deque可用
vector<int> vec;
vec.push_back(2);
vec.push_back(4);
vec.push_back(-1);
sort(vec.begin(), vec.end(), cmp);
// 排序结构体
struct node {
  int x, y;
} ssd[10];
bool cmp_stru(node n1, node n2) {
  return n1.x > n2.x;
}
sort(ssd, ssd+4, cmp_stru);
```

`lower_bound()`返回第一个*等于或者大于*目标元素的指针（数组）或者迭代器，如果没找到返回适合插入该元素的位置；

`upper_bound()`返回第一个*大于*目标元素的指针（数组）或者迭代器，如果没找到返回适合插入该元素的位置；

因此，如果没有找到对应元素，两个函数会返回相同的值

```c
int * low_pos = lower_bound(a, a+2, 3);
int * up_pos = upper_bound(a, a+2, 3);
printf("lower_bound: %d\n", low_pos - a); // 返回下标
```

