---
title: 2-basic
notshow: false
date: 2021-12-04 17:08:38
categories:
- algorithm
tags:
- book
---

# C语言基础要点

《算法笔记》第二章 C/C++快速入门。这里记录些要点。

C语言常包括头文件`<stdio.h>`，`stdio`是标准输入输出的意思。实际上，在c++标准中，推荐使用`<cstdio>`，`cmath`和`cstring`等头文件，和`.h`结尾的头文件是等价的。

<!--more-->

## 基本数据类型

int的取值范围在$10^9$范围内，long long的范围在$10^{18}$。如果要给int定义一个表示无穷大的数，推荐取值为$2^{30}-1$，这样能够避免相加后超出int取值范围，一般定义是`int INF=(1<<30)-1`。

浮点数float的精度是6-7位，double是15-16位，推荐一般使用double。

单个char赋值的时候，字符常量只有单个字符，要使用`''`单引号。字符编码记住小写字母>大写字母>数字，大写字母+32就是对应的小写字母ASCII码。

字符串可以使用char数组，`char str[24]`，输入和输出可以直接使用`%s`。字符串一定不能赋值给char。

条件运算符是c语言中的唯一三母运算符，`int c = a>b ? a : b;`。

## scanf与printf

scanf和printf比c++中的cin和cout要快，在某些要求时间约束的题目中好用。

输入和输出long long，都使用`%lld`。字符是`%c`，输入float是`%f`，输入double是`%lf`，输入字符串是`%s`。记住`%c`可以直接获得空格和换行符，`%s`遇到空格和换行符就会停止输入。

scanf的格式：`scanf("格式控制",  变量地址)`。注意除了`%s`由于是数组首元素地址，其它变量都需要使用`&`获得变量地址。

printf的格式：`printf("格式控制"，变量名)`。printf输出变量时，除了double和float一样都使用`%f`即可，其它都和scanf一样。几个特殊的控制输出的格式：

- `%md`：`m`是控制输出整型以m位右对齐，如果整型变量本身超出m位，就保持原样。
- `%0md`：对于左边不够m位的，使用`0`补齐
- `%.mf`：浮点数保留m位，保留的规则不是简单的四舍五入，而是四舍六入五成双。
- `%%`和`//`：输出`%`和`/`。

`getchar()`和`putchar(char)`可以方便的获取单个字符。

## 常用math函数

下面提到的math函数，输入都是`double`。

- `fabs(double x)`：double取绝对值
- `floor(double x)`和`ceil(double x)`：向下和向上取整，注意负数的向下取整是取更负的值。
- `pow(double r, double p)`：计算$r^p$。
- `sqrt(double x)`：返回算术平方根
- `log(double x)`：返回$e$为底的对数值，如果要计算非$e$为底的对数，利用换底公式$log_ab=log_eb/log_ea$。
- `sin(double x), cos(), tan()`：输入的`x`是弧度，不是角度，注意`弧度=角度 x pi/180`。
- `asin(double x), acos(), atan()`：反三角函数，返回弧度值
- `round(double x)`：四舍五入

## 数组

### 一般数组

数组的一般定义形式

```c
数据类型 数组名[数量]
```

*注意数组传入函数时，如果是二维数组，需要指定第二维的长度，如果是一维数组不要求指定；在函数中修改数组元素，会直接修改原数组的值*。

定义的时候初始化：

```c
// 一维数组
int a[45] = {1, 2, 3, 5};
// 二维数组
int b[2][3] = {{0, 1, 2}, {}, {2, 4}};
// 字符数组
char c[2] = {'a', 'c'};
char c[2] = "ac"; // 注意只有定义的时候可以这样用""定义
```

注意，如果定义的数组较大（元素数量>$10^6$），由于函数内部的局部变量来自系统栈，允许的空间较小，需要移到函数外，函数外的全局变量来自静态存储区，允许申请的空间比较大。

利用函数初始化数组，有两种函数：

- `memset(数组名, 值, sizeof(数组名)`：按照*字节*依据所给的值赋值给数组元素，因此为了避免错误，一般是使用`0`或者`-1`初始化。该函数执行速度较快。需要`#include<string>`
- `fill(数组开始地址, 数组结束地址, 值)`：需要包括`#include<algorithm>`，便于利用其他值初始化数组，不会出错，执行速度更慢。

### 字符数组

字符数组的输入与输出：

1. 使用`scanf`和`printf`

```c
scanf("%s", str); // 遇到空格或者换行停止
printf("%s", str);
```

注意，使用scanf输入时，编译器会自动在字符串末尾添加`\0`，因此字符串数组大小至少要比规定的大小大1。使用printf输出时，识别`\0`作为中断输出，如果没有该字符，会输出乱码。

2. 使用`getchar`和`putchar`

输入输出单个字符，注意该方法无法自动识别字符串尾端，需要人工增加`\0`避免出错。

```c
char c = getchar();
putchar(c);
```

3. 使用`gets`和`puts`

用来直接输入和输出字符串，记住`gets(str)`是以`\n`作为输入结束标志，因此`gets()`可以用来输入有空格的字符串，它也会自动添加`\0`。

```c
char str1[100];
gets(str1);
puts(str1);
```

`string.h`头文件包含了很多有用的处理字符串的函数：

- `strlen(str)` ：返回字符个数，`int len=strlen(str);`。
- `strcmp(str1, str2)`：按照字典序比较字符串，`str1<str2`返回负数，`str1==str2`返回`0`，`str1>str2`返回正数。

- `strcpy(str1, str2)`：拷贝str1给str2。
- `strcat(str1, str2)`：把str2连接到str1后面，`strcat(str1, str2);`。

介绍两个在`strio.h`中就包含的函数，`sscanf`和`sprintf`。`sscanf`用来把一个字符串按照要求的格式赋值给其它变量，`sprintf`用来把其它变量按照要求的格式赋值给字符串。

```c
char str[50]="12345";
int n;
// sscanf用法
sscanf(str, "%d", &n);
char str1[50]="2048:3.14,hello";
double b;
char s[10];
sscanf(str1, "%d:%lf,%s", &n, &b, s);
// sprintf用法
sprintf(str, "%d:%.1lf", n, b); // 此时str由12345变为2018:3.1
```

## 指针

在c语言中，指针是指向变量首个字节地址的变量，由于不同类型的变量有不同的字节，因此还需要确定变量的类型。`int *p = &a;`。指针本身都是一个unsigned的int。

如果定义多个指针，都需要包括`*`，`int *p1, *p2, *p3;`。

指针变量支持自增和自减操作。

c语言中的数组名就代表首个数组元素的地址，也可以直接看做是个指针。因此

```c
int a[10];
int *p = a;
scanf("%d", a); // 会直接赋值给a[0]
scanf("%d", a+1); // 会直接赋值给a[1]
```

指针作为函数参数传入时，指针参数本身是值传递，但是通过指针操作被指向的变量可以直接修改。

```c
void change(int *p) {
  *p = 1; // 会修改被指向变量的值
}
```

c++提供了引用的语法，用在函数的形式参数中，用来表示该形式参数只是传入变量的一个别名，不会拷贝和复制，这样不需要指针也能够直接修改传入的原参数。

```c
void swap(int &a, int &b) {
  int c = a;
  a = b;
  b = c;
}
int x = 0, y = 1;
swap(x, y); // 注意，x和y不能再加&
```

## 结构体的使用

```c
struct studentInfo {
  // 基本数据类型
  int age;
  char name[10];
  studentInfo * nextStu;
} stu, *stuP, students[20];
```

注意，定义的时候，结构体内不能有自身类型，但是可以有指向自身的指针。

访问结构体元素：

- 非指针：`stu.age; stu.name`

- 指针：`stuP->age; stuP->name`

构造函数，为了方便直接利用已有的基本数据类型变量生成一个结构体实例，c语言提供了结构体的构造函数，没有返回类型，构造函数命名就是本身。

```c
struct studentInfo {
  // 基本数据类型
  int age;
  char name[10];
  studentInfo * nextStu;
  // 构造函数
  // 默认构造函数，可以设计多个应用与不同场景，只要形参不同即可
  studentInfo() {} // 便于不初始化也能定义结构体变量
  studentInfo(int _age, char _name[]) {
    age = _age;
    strcpy(name, _name);
  }
  studentInfo(int _age) {
    age = _age;
  }
};
studentInfo stu; // 如果没有默认构造函数就无法这样定义
studentInfo stu2 = studentInfo(18, "alexpp");
```

## 补充

cin读入字符串数组的时候，直接`cin>>str`即可，但是如果希望读入一整行的话，使用`getline`，比如`getline(cin, str)`，这个方法对于STL中的string同样适用。

cout的输出，如果需要指定浮点数的精度，需要包括`<iomanip>`，

```c
cout<<setiosflags(ios::fixed)<<setprecision(2)<<123.456<<endl;
```

浮点数的比较，在进行了可能影响小数点的精度的运算之后，本来两个从原理上应该相等的浮点数，结果由于舍弃了部分小数位，导致不相等，这时候需要考虑误差允许范围内的比较操作。

定义一个小数，`const double eps = 1e-8 `，`1e`代表的是`10`。

重新定义`==, !=,<,>,>=,<=`。

举例：

```c
bool equ(double a, double b) {
	return fabs(a - b) < eps;
}
bool large(double a, double b) {
  return (a - b) > eps;
}
bool less() {
  return (b - a) > eps;
}
bool largeEqu() {
  return a > (b - eps);
}
bool lessEqu() {
  return a < (b + eps);
}
```

计算$\pi$，使用`const double pi = acos(-1.0);`即可。

## 黑盒测试

单点测试就是OJ对于每组输入都重新启动文件，程序运行一次只需要处理一组测试数据，即程序只需要保证单次运行成功即可。

多点测试就是要求程序必须一次运行所有组测试数据，根据题目不同有不同写法。

由于OJ是把所有测试数据放在一个文件中，因此只要判断测试文件是否已经输入完毕即可。`scanf`函数会返回成功读入的参数的个数，如果读到文件末尾就会返回`-1`，在c语言中使用`EOF`代表-1。

```c
while(scanf("d", &n)!=EOF) {
  // pass
}
// 如果是在读入字符串
while(scanf("s", str)!=EOF) {
  // pass
}
while(gets(str)!=NULL) {
    // pass
}
```

当然根据题目，可能还有其它合适的循环读入的方法。
