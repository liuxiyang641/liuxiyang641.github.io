---
title: 12-string
notshow: false
date: 2021-12-04 14:39:12
categories:
- algorithm
tags:
- book
---

# 字符串专题

<!--more-->

## 字符串hash进阶

在前面第4章讨论过对只有大小写字母的字符串的hash办法，简单说就是把字母看出26进制，然后换算成10进制
$$
H[i] = H[i-1]\times26 + index(str[i])
$$
为了避免结果太大，取模
$$
H[i] = \left(H[i-1]\times26 + index(str[i])\right)\ \%\ mod
$$
当然这可能出现hash碰撞。幸运的是，在实践中发现，如果选择合适的进制和取模除数，可以很大程度避免这个问题，一般来讲，设置进制`p`为一个$10^7$的素数（如10000019），`mod`为$10^9$的素数（例如1000000007），冲突概率就非常小了。
$$
H[i] = \left(H[i-1]\times p + index(str[i])\right)\ \%\ mod
$$
字符串的子串hash问题：如何表示`H[i...j]`？

思路：我们当然可以对子串同样进行相同的hash操作，但是这样会造成大量的冗余计算。我们可以利用上面计算的结果来简化操作，`H[j]`可以由`H[i-1]`一路推导下来：
$$
H[j] = H[i-1]\times p^{j-i+1} + H[i...j] \\
H[i...j] = H[j] - H[i-1]\times p^{j-i+1}
$$
如果加入取模操作，注意`H[j]`可能会小于`H[i-1]xp`，直接取模可能得到负值，因此为了得到非负结果，结果取模后再度加一次`mod`，然后再取模，保证能够得到正值。
$$
H[i...j] = ((H[j] - H[i-1]\times p^{j-i+1})\%\ mod + mod)\%\ mod
$$
因此，我们可以在计算获得`H[i]`之后，按照上面的式子方便的获得所有子串的hash值，用于之后的计算。

可以用于计算两个字符串的最大公共子串（注意不是子序列），只需要比较两个子字符串的所有子串hash是否相同，并取最大值即可；

也可以用于计算字符串的最大回文子串，把原来的字符串反转，然后比较最大的hash相同的公共子串。

最后，如果出现了hash冲突，只需要改变下`p`和`mod`即可。甚至还可以采用双hash，也就是计算两个不同的hash值一起表示字符串的办法。

## KMP算法

接下来讨论字符串匹配问题，KMP算法是三个发明作者的首字母。

思想：核心是设计一个next数组，`next[i]=k`表示字符串`s[0..i]`的最长相等前缀`s[0...k]`和后缀`s[i-k...i]`。

获取next数组的代码：

```c
void getNext(char s[], int len) {
  int j = -1;
  next[0] = -1; // -1表示没有匹配的前后缀
  for(int i=1; i<len; ++i) {
    // 后缀最后一位一定是s[i]，因此前缀最后一位必须匹配s[i]
    while(j!=-1 && s[i]!=s[j+1]) {
      j = next[j];
    }
    // 如果成功匹配
    if(s[i]==s[j+1]) {
      j++;
    }
    next[i] = j; // 设置最大匹配前后缀
  }
}
```

KMP算法就是利用前面的next数组来实现，对于字符串`text`和模式串`pattern`。计算`pattern`的next数组，然后模仿next的计算过程不断去匹配`text`。实际上，`next`数组的求解过程，就是自身对自身的匹配。

```c
// 返回匹配的字符串个数
int KMP(char text[], char pattern[]) {
  int n = strlen(text), m = strlen(pattern);
  getNext(pattern, m);
  int j = -1; // 此时j表示在pattern上的位置
  int matchCount = 0;
  for(int i=0; i<n; ++i) {
    // 没有办法继续匹配，就让pattern已匹配的前缀后退
    while(j!=-1 && text[i]!=pattern[j+1]) {
      j = next[j];
    }
    if(text[i] == pattern[j+1]) {
      j++;
    }
    // 如果已经全部匹配
    if(j==m-1) {
      matchCount++;
      j = next[j]; // 回退，然后继续尝试匹配
    }
  }
  return matchCount;
}
```

