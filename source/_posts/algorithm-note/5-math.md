---
title: 5-math
notshow: false
date: 2021-12-13 10:29:31
categories:
- algorithm
tags:
- book
---

# 数学问题

《算法笔记》第五章 数学问题

<!--more-->

## 最大公约数和最小公倍数

求解最大公约数的方法一般为欧几里得法，即辗转相除法。0和任意整数的最大公约数都是整数本身。

```c
int gcd(int x, int y) {
	if(y==0) return x; // 保证除数不是0
  return gcd(y, x % y);
}
```

最小公倍数的求解方法是在求解最大公约数的基础上进行的，最小公倍数的求解方法未`a*b/d`，`d`就是最大公约数。为了避免超出类型范围，一般为`a / d * b`。

## 素数

素数，也叫做质数是指除了1和本身之外，不能被任意的整数整除的数字。

合数是指可以被除了1和本身之外的数字整除的数字。

特殊的是1，*1既不是素数，也不是合数*。

### 判断素数

假设数字n可以被k整除：`n % k == 0`，我们当然可以遍历从2开始到n-1的所有值检查是否可以整除。

实际上，可以缩小判断范围，考虑平方根，`sqrt(n)*sqrt(n) == n`，所有可以整除n的第一个数字，一定是`<= sqrt(n)`，因此我们只需要遍历从2开始到sqrt(n)。

```c
bool isPrime(int x) {
  int sqr = sqrt(x);
  for(int i = 2; i <= sqr; ++i) {
    if(n % i == 0) {
      return false;
    }
  }
  return true;
}
```

### 获取素数表

我们当然可以判断每一个数字是否是素数获得素数表，但是这样效率较低。

可以考虑使用埃式筛选法：

如果一个数字`i`是素数，就把它所有的后续整倍数筛选出去。如果一个数没有被筛选出去，那它就是素数。初始化默认2是素数。

```c
int prime[maxn];
int pNum = 0;
bool isPrime[maxn] = {true};
void findPrime() {
  for(int i = 2; i < maxn; ++i) {
		if(isPrime[i]) {
      prime[pNum++] = i;
      for(int j = 2; i * j < maxn; ++j) {
				isPrime[i * j] = false;
      }
    }
  }
}
```

## 质因子分解

把一个数字`n`分解为多个质数的乘积形式。

首先通过获取质数表，我们可以获得所有的候选质数，然后判断各个质数是否是数字`n`的因子。

同样的，类似于判断质数的方法，数字`n`的`>=sqrt(n)`的质因子至多有一个，`<sqrt(n)`的质因子可以有多个，因此，我们可以先寻找所有`<sqrt(n)`的质因子，之后如果发现乘积不为数字`n`，则继续计算。

```c
struct factor {
  int x, cnt; // 记录质因子和个数
} fac[10]; // 10个最多了
findPrime();
int sqr = sqrt(n);
int num; // 记录所有质因子的数量
// 遍历素数表
for(int i = 0; i < pNum && primes[i] <= sqr; ++i) {
  if(n % prime[i] == 0) {
    fac[num].x = prime[i];
    fac[num].cnt = 0;
    while(n % prime[i] ==0) {
      n = n / prime[i];
      fac[num].cnt++;
    }
    num++;
  }
  if(n==1) break; // 已经找到所有的质因子
}

if(n != 1) {
  // 如果还存在>=sqrt(n)的质因子
  fac[num].x = n;
  fac[num].cnt = 1;
}
```

## 大整数运算

### 大整数的存储

对于过于大的整数，比如1000位的整数，不能再使用基本类型存储，因此考虑使用结构体进行存储。

```c
struct bign {
  int d[1000]; // 保存整数的各位数字，[0]是最低位
  int len; // 保存位数
  bign() {
    memset(d, 0, sizeof(d)); // 使用0初始化所有位，方便四则运算
    len = 0;
  }
}
```

需要记住我们使用从低位到高位的存储办法。因此，在读入大整数的时候，对于读入的大整数可以先考虑使用`char[]`保存，然后逆位赋值给`bign`。同理，`bign`和`bign`的比较，同样是从`len-1`开始比较。

### 大整数的四则运算

大整数的加法，从最低位开始相加，如果大于10就进位（除以10），余数在当前位。

```c
bign add(bign a, bign b) {
  bign c;
  int carry = 0;
  int tmp;
  // 从低位开始加起
  for(int i = 0; i < a.len || i < b.len; ++i) {
    tmp = a.d[i] + b.d[i] + carry;
    c.d[i] = tmp % 10;
    c.len++;
    carry = tmp / 10;
  }
  if(carry != 0) { // 加法的最终进位最多1位
    c.d[len++] = carry;
  }
  return c;
}
```

大整数的减法，与加法的一个很大区别是，需要首先判断两个大整数的大小，总是使用大整数减去小整数，对于结果是负数的情况额外输出负号即可。

```c
bign sub(bign a, bign b) {
  bign c;
  for(int i =0; i < a.len || i < b.len; ++i) {
    if(a.d[i] - b.d[i] < 0) {
      a.d[i + 1]--;
      a.d[i] += 10;
    }
    c.d[len++] = a.d[i] - b.d[i];
  }
  // 减去高位可能存在的多个0，但是至少保留1位，例如3333332-3333331=1
  while(c.len > 1 && c.d[len - 1] == 0) {
    c.len--;
  }
  return c;
}
```

大整数与int的乘法，类似于大整数的加法，从低位到高位，将int与大整数的某一位相乘，结果加上进位，然后个位保留作为该位结果，更高位作为进位。

```c
bign mult(bign a, int b) {
  bign c;
  int tmp, carry = 0;
  for(int i = 0; i < a.len; ++i) {
    tmp = a.d[i] * b + carry;
    c.d[len++] = tmp % 10;
    carry = tmp / 10;
  }
  while(carry != 0) {
    c.d[len++] = carry % 10;
    carry /= 10;
  }
  return c;
}
```

大整数与int的除法，每一步都是上一步的余数乘以10，加上当前位与除数相除，结果作为当前位，余数留到下一位。除法需要从高位开始操作。

```c
bign divide(bign a, int b) {
  bign c;
  c.len = a.len; // 余数的位数最多和被除数一样
  int tmp, carry = 0;
  for(int i = a.len - 1; i >= 0; --i) {
		tmp = carry * 10 + a.d[i];
    a.d[i] = tmp % b;
    carry = tmp / b;
  }
  // 去除高位的0
  while(c.len > 1 && c.d[len - 1] == 0) {
    c.len--;
  }
  return c;
}
```

## 组合数

关于问题，求`n!`有几个质因子`p`？

可以记住一个式子，`n!`中有$\frac{n}{p}+\frac{n}{p^2}+\frac{n}{p^3}+\dots$的质因子`p`。

```c
int cal(int n, int p) {
	int ans = 0;
  while(n) {
    ans += n / p;
    n /= p;
  }
  return ans;
}
```

上面答案的一个变形是求解`n!`会有末尾0的个数，本质上等于求有多少个2和5的组合的个数，又因为质因子2的数量多于5，因此直接求解`n!`有多少个质因子5即可。

求解组合数$C_{n}^{m}$，如果直接使用公式$\frac{n!}{m!(n-m)!}$可能超界限，可以考虑使用公式$C_{n}^{m}=C_{n-1}^{m-1}+C_{n-1}^{m}$。

```c
long long res[67][67] = {0};
long long C(long long n, long long m) {
  if(m==0 || n==m) {
    return 1;
  }
  if(res[n][m] != 0) {
    return res[n][m];
  }
  return res[n][m] = C(n-1, m) + C(n-1, m-1);
}
```

