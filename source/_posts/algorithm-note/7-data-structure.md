---
title: 7-data-structure
notshow: false
date: 2021-11-24 19:18:31
categories:
- algorithm
tags:
- book
---

# 第七章 数据结构专题

《算法笔记》笔记。

<!--more-->

## 栈的应用

这里记录下使用数组实现栈的思路，核心是使用一个栈顶指针记录位置。

```c
int s[100000]; // 栈的定义
int TOP = -1; // 栈顶指针
// 自定义的常用函数，可以看到实现非常简单，主要是直接对TOP指针的操作
void clear() {
  TOP = -1;
}

int size() {
	return TOP + 1;
}

void push(int x) {
  s[TOP++] = x;
}

void pop() {
  TOP--;
}

int top() {
  return s[TOP];
}

bool empty() {
  if (TOP == -1) return true;
  return false;
}
```



## 队列的应用

这里同样使用数组实现队列，和实现栈不一样的是，这里维护两个指针`front`和`back`。

```c
int q[100000];
int front = -1, back = -1; // 队首与队尾指针，让队首始终指向首个元素的前一位，这样方便判断只有一个元素时，队列是否为空
// 常用函数
void clear() {
  front = back = -1;
}
int size() {
  return back - front;
}
bool empty() {
  if (front == back) return true;
  return false;
}
int push(int x) {
  q[++back] = x;
}
void pop() {
  front++;
}
int get_front() {
  return q[front+1];
}
int get_back() {
  return q[back];
}
```

## 链表的处理

这里讨论的链表，类似于vector，只不过是自己实现。

### 动态链表

先讨论动态链表，即动态生成、删除、释放链表节点。

首先需要定义节点用来存储数据，

```c
struct Node {
  typename data;
  Node * next;
}
struct Node {
  int data;
  Node * next;
}
```

接下来讨论如何动态生成新节点，在c语言和c++中都有不同的实现办法，c语言中使用`malloc()`函数，c++中使用`new`操作符。

```c
// 使用malloc函数，malloc函数会返回(void *)的指针，使用(typename*)进行类型转化
#include<stdlib.h>
typename * p = (typename *)malloc(sizeof(typename));
// 举例
Node * p = (Node *)malloc(sizeof(Node));
// 在c++中直接使用new
typename * p = new typename;
// 举例
Node *p = new Node;
```

如何释放新节点？

为什么要释放新节点？

*因为c语言的设计者认为程序员完全有能力自己控制内存的分配与释放*。

c语言中使用`free()`函数，c++中使用`delete()`。

```c
// c语言
free(p);
// c++
delete(p);
```

链表的创建，首先第一个问题是是否拥有头结点（dummy node）？这里按照《算法笔记》上的内容，默认拥有头结点，头结点不是第一个节点，头结点的`next`指向第一个保存数据的结点，头结点本身不储存任何数据。这样的好处之一是无论链表是否有数据，总会有一个固定的头结点，方便判断链表是否为空，以及插入新的结点等。

尾结点的`next`应该设置为`NULL`。

对于动态链表的各项操作，这里不写出来。

### 静态链表

静态链表和动态链表的区别是，静态链表是通过定义数组的方式提前开辟好的连续内存，`next`就是下一个结点在数组中的下标。

静态链表的适用范围是在需要的地址比较小的时候，比如`<10^5`。

```c
// 静态链表的定义
struct Node {
  int data;
  int next; // next == -1时表示到了链表尾端
  bool is_node; // 这里是用来记录当前的数组元素是否属于链表的一个节点，默认应该设置为false
}
Node p[100010];
// 访问下个结点
int begin_address = 2210;
int next_address = p[begin_address].next;
printf("%d", p[next_address].data);
// 静态链表的好处是，便于直接适用sort函数实现排序
bool cmp (Node n1, Node n2) {
  if(!n1.is_node || !n2.is_node) {
    return n1.is_node > n2.is_node; // 让是结点元素的在数组靠前的位置
  }
  return n1.data < n2.data; // 让data小的在前
}
```

