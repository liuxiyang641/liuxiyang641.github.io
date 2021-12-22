---
title: 9-data-structure2
notshow: false
date: 2021-11-26 10:23:33
categories:
- algorithm
tags:
- book
---

# 数据结构专题2

《算法笔记》第九章，数据结构专题2，主要涉及树、并查集、堆等。

<!--more-->

## 树和二叉树

树的基本定义：在数据结构中的树只有一个根结点，若干个节点和边，并且所有的节点是连通的且无闭环。

几个比较实用的性质：

- 树可以是空的，此时连根结点都没有，叫做空树。
- 树的层次layer，是从根结点作为第一层开始计算的，依次递增。
- 结点的子树数量叫做度degree，注意不计算空树。结点的最大度数看做是树的度数，叫做宽度。
- 树中的边只能连接两个结点，因此，对于有$n$个结点的树，一定有$n-1$个边。如果有$n$个相互连通的结点，并且有$n-1$个边，那么一定是一棵树。
- 叶子结点是度为0的结点。
- 节点的深度是从根结点深度为1开始算起到该节点的层数；节点的高度是从最底层的叶子结点到该节点的层数。类似于树的宽度定义，树的深度是结点的最大深度，树的高度是根结点的高度。
- 多棵树组合在一起叫做森林forest。

二叉树的递归定义：

1. 二叉树可以是空树（没有根结点）；
2. 二叉树由根结点、左子树、右子树组成。左右子树都是二叉树；

二叉树一定是度为2的树，但是度为2的树不一定是二叉树，因为二叉树的左右子树是严格区分的，不能互换。

两种特殊的二叉树：

1. 满二叉树：除去叶子结点外，所有的节点的左右子树都是满的（即树的每一层都是满的，包括最底层）
2. 完全二叉树：除去最底层外，所有层都是满结点的；并且最底层的叶子结点从左到右是连续的。

二叉树的存储，一般的二叉树使用链表来定义（对于完全二叉树可以使用数组来定义）：

```c
struct Node {
	typename data;
  Node * lchild; // lchild == NULL表示左子树为空
  Node * rchild; // rchild == NULL表示右子树为空
}
// 定义一棵空树
Node * root = NULL;
```

新建一个结点，但是还未加入到树中，具体要增加到树的哪个地方依赖于之后的具体二叉树的性质。

```c
Node * newNode (int data) {
  // c语言和c++两种创建方式
  Node * node = new Node;
  Node * node = (Node*)malloc(sizeof(Node));
  node->data = data;
  node->lchild = node->rchild = NULL;
  return node;
}
```

二叉树的查找与修改：查找到所有满足查询条件的结点，并且可以修改数据

```c
// 下面的代码是从左开始的DFS
void search(Node* current_root, int x, int new_data) {
  if (current_root==NULL) return; // 空树返回
  if (current_root->data == x)
    current_root->data = new_data;
  // 继续查询左子树
  search(current_root->lchild, x, new_data);
  // 查询右子树
  search(current_root->rchild, x, new_data);
}
```

二叉树的插入依赖于具体二叉树的性质，但是一般来说，二叉树的插入就是在不满足某种条件的位置，并且这个位置一般是确定唯一的，否则就会有多个位置适合插入了。

下面是一个二叉树插入新结点的伪代码：

```c
// 下面的函数需要特殊注意，current_root是引用形式，这样方便直接修改传入的root值
void insert(Node* &current_root, int new_data) {
  if(current_root==NULL) {
		root = newNode(new_data); // 没有找到合适位置，就是目前的空树插入新结点
    return;
  }
  
  if (根据new_data和目前节点的数据判断发现应该插入到左子树) {
    insert(current_root->lchild, new_data);
  }
  else { // 根据new_data和目前节点的数据判断发现应该插入到右子树
    insert(current_root->rchild, new_data);
  }
}
```

二叉树的创建，就是重复上面的插入函数：

```c
Node* create(int data[], int n) {
  Node* root = NULL; // 注意不能new Node，这样就不是空树了
  for(int i = 0; i < n ; i++) {
    insert(root, data[i]);
  }
  return root;
}
```

上面是一般的二叉树存储，对于完全二叉树，在元素数量不多的情况下，可以直接使用数组存储。

将根结点放在数据的位置$1$，不能从$0$开始存放，否则会丧失下面的规律：

对于结点$i$，它的左子树位置一定是$2i$，右子树位置是$2i+1$。

判断是否是叶子结点，该结点的左子树位置超过了树的结点数量$n$。

## 二叉树的遍历

这里讨论二叉树的四种遍历方法：

- 先序遍历：根结点、左子树、右子树
- 中序遍历：左子树、根结点、右子树
- 后序遍历：左子树、右子树、根结点
- 层序遍历：BFS，访问同一层的结点结束后，再访问下一层

在上面的遍历过程中，可以发现，总是先访问左子树，再访问右子树。

下面是先序遍历的函数

```c
void preSearch(Node* root) {
  if(root==NULL) return;
  printf("%d ", root->data);
  preSearch(root->lchild);
  preSearch(root->rchild);
}
// 中序与后序遍历就是上面代码的顺序改变下即可
```

层序遍历，利用到之前学的使用队列实现BFS

```c
void layerSearch(Node* root) {
  queue<Node*> q; // 注意是结点指针的队列
  q.push(root);
  while(!q.empty()) {
    Node* tmp_node = q.front();
    q.pop();
    printf("%d ", tmp_node->data);
    // 左右子树入队
    q.push(tmp_node->lchild);
    q.push(tmp_node->rchild);
  }
}
```

先序、中序以及后序遍历的作用在于，可以根据先序+中序或者后序+中序的输出，来唯一的重建树。先序或者后序都可以确定当前序列的根结点（第一位或者最后一位），然后结合中序就可以找到左右子树。

利用先序+中序重建树，如果两者对应的序列数组是$[preL, preR]$和$[inL, inR]$，那么根结点是$preL$位置的元素，根据根结点寻找根结点在中序中的位置假设为$k$，那么对应的左子树的先序和中序序列是$[preL+1, preL+k-inL]$，$[inL, k-1]$；右子树的先序序列是$[preL+k-inL+1, preR]$，中序序列是$[k+1, inR]$。

使用递归的方式就可以实现重建树。

二叉树也可以使用数组的方式来实现，指针变为数组的下标，左右子树都指向数组的下标，同时，如果是空树就设为-1，这里不赘述具体实现。

## 树的遍历

一般的树有多个子结点，并且不限制左右树顺序。

对于多个子树，当然可以使用链表来保存，但是有些麻烦，这里按照书上的说明统一使用静态写法，

```c
struct Node {
  typename data;
  vector child;
} node[maxn];
```

创建新结点的办法，类似与前面的静态二叉树：

```c
int index = 0; // 记录当前的新元素在tree数组中的位置，index下没有数据
int newNode(int x) {
	node[index].data = x;
  node[index].child.clear();
  return index++;
}
```

## 二叉查找树（BST）

二叉查找树就是满足左子树中的元素都$<=$根结点，右子树都$>$根结点的二叉树。当然，$==$根结点的元素到底在哪颗子树存放可以看题目的具体要求。

bst的查找：

```c
Node* search(Node* root, int x) {
  if(root==NULL) return NULL;
  if(x==root->data) {
    return root;
  }
  if (x<root->data) return search(root->lchild, x);
  else return search(root->rchild, x);
}
```

bst的插入：

```c
void insert(Node* &root, int x) {
  if(root==NULL) {
    node = new Node;
    node->data = x;
    node->lchild = node->rchild = NULL;
    root = node;
  }
  if (x<=root->data) insert(root->lchild, x);
  else insert(root->rchild, x);
}
```

bst的删除比较特殊，主要是涉及如何让结点被删除后，以该结点为根结点的整棵树还能够保证原来二叉查找树的性质。

有两种办法：用该结点的前驱或者后继来替换该结点。

- 前驱：某个节点左子树中的最大结点，在树上表现为左子树的最右结点（可能有自己的左子树）
- 后继：某个结点右子树中的最小结点，在树上表现为右子树的最左结点（可能有自己的右子树）

寻找前驱和后继：

```c
// 寻找某棵树的最大结点
Node * findMax(Node* root) {
  while(root!=NULL) {
    root = root->rchild;
  }
}
Node* preMaxNode = findMax(root->lchild);
// 寻找某棵树的最小结点
Node * findMin(Node* root) {
  while(root!=NULL) {
    root = root->lchild;
  }
}
Node* postMinNode = findMin(root->rchild);
```

bst的删除操作：

```c
void deleteNode(Node* &root, int x) {
  if(root==NULL) return;
  if(root->data == x) {
    // 如果要删除的结点是叶子结点
    if(root->lchild == NULL && root->rchild==NULL) {
      delete(root);
      root=NULL;
    }
    // 寻找前驱
    if(root->lchild != NULL) {
      Node* preMaxNode = findMax(root->lchild);
      root->data = preMaxNode->data; // 替换为前驱
      deleteNode(preMaxNode, preMaxNode->data); // 删除前驱
    }
    else {
      Node* postMinNode = findMin(root->rchild);
      root->data = postMinNode->data; // 替换为后继
      deleteNode(postMinNode, postMinNode->data); // 删除后继
    }
  }
  else if(root->data < x) {
    deleteNode(root->rchild, x);
  }
  else {
    deleteNode(root->lchild, x);
  }
}
```

##平衡二叉树（AVL）

平衡二叉树是对于前面二叉查找树的改进，“平衡“的意思是保证左右子树的高度差不超过1，这样总能够保证找寻所需元素时的搜索效率在$O(log(n))$内。

平衡二叉树是两位前苏联的数学家提出来的，取他们名的大写字母命名为AVL树。在AVL树中的每个结点会额外记录一下当前结点的高度，同时利用左右子树的结点高度差$左子树高度-右子树高度$，可以可以计算当前结点的*平衡因子*。

AVL树的难点在于插入与删除，一个结点的定义

```c
struct Node {
  int data;
  int height; // 默认为1
  Node* lchild, rchild;
}
```

新建结点

```c
Node* newNode(int x) {
  Node* node = new Node;
  node->data = x;
  node->height = 1; // 新增的定义
  node->lchild = node->rchild = NULL;
  return node;
}
```

查询当前结点的高度与计算平衡因子

```c
int getHight(Node* node) {
  return node->height;
}
int getBalanceFactor(Node * node) {
  return getHeight(node->lchild) - getHeight(node->rchild);
}
```

更新节点高度

```c
void updateHeight(Node* node) {
  node->height = max(getHeight(node->lchild), getHeight(node->rchild)) + 1;
}
```

AVL树的查询操作，与一般的BST树一样，这里不重复叙述。

AVL树的插入操作，比较复杂，需要考虑在插入新结点后，如何保持”平衡“？这里有两种办法调整当前树的左右子树高度，*左旋*和*右旋*：

- 左旋：通过旋转，让当前根结点变为原来右子树根结点的左结点，原来右子树根结点成为新根结点
- 右旋：通过旋转，让当前根结点变为原来左子树根结点的右结点，原来左子树根结点成为新根结点

详细的旋转过程可以参考《算法笔记》上的图示，这里提供代码，核心思想在于：

- 左旋：当前根结点变为新的左结点，当前根结点的右子树链接到右结点（新根结点）的左子树
- 右旋：当前根结点变为新的右结点，当前根结点的左子树链接到左结点（新根结点）的右子树

代码：

```c
// 左旋
void L(Node* &root) {
  Node * tmp = root->rchild; // 新根结点是原右子树
  root->rchild = tmp->lchild; // 当前根结点右子树变为原右子树的左子树
  tmp->lchild = root; // 新根结点的左子树是原根结点
  updateHeight(root); // 先更新新树的子树
  updateHeight(tmp); // 然后更新根结点
  root = tmp; // 将root改为新的根结点
}
// 右旋
void R(Node* &root) {
  Node * tmp = root->lchild; // 新根结点是原左子树
  root->lchild = tmp->rchild; // 当前根结点左子树变为原左子树的右子树
  tmp->rchild = root; // 新根结点的右子树是原根结点
  updateHeight(root); // 先更新新树的子树
  updateHeight(tmp); // 然后更新根结点
  root = tmp; // 将root改为新的根结点
}
```

接下来可以讨论AVL树插入新元素的操作了。当AVL树插入新结点后，可能造成左子树比右子树高$2$，或者是右子树比左子树高$2$。一共有四种可能（更详细的讨论在书上）：

1. LL型：左子树高->左子树的左子树高；右旋根结点解决
2. LR型：左子树高->左子树的右子树高；先左旋左子树，后右旋根结点
3. RR型：右子树高->右子树的右子树高；左旋解决
4. RL型：右子树高->右子树的左子树高；右旋右子树，后左旋根结点

插入操作的代码：

```c
void insert(Node* &root, int x) {
  if (root==NULL) {
    root = newNode(x);
    return;
  }
  if(root->data > x) { // 插入到左子树
    insert(root->lchild, x);
    updateHeight(root);
    if(getBalanceFactor(root) == 2) { // 如果左子树失衡了，因为是递归，总能保证从下往上调整失衡树
			if(getBalanceFactor(root->lchild) == 1) { // LL型
        R(root);
      } else if(getBalanceFactor(root->lchild) == -1) { // LR型
        L(root->lchild);
        R(root);
      }
    }
  }
  else { // 插入右子树
    insert(root->rchild, x);
    updateHeight(root);
    if(getBalanceFactor(root) == -2) { // 如果左子树失衡了
			if(getBalanceFactor(root->rchild) == 1) { // RL型
        R(root->child);
        L(root);
      } else if(getBalanceFactor(root->rchild) == -1) { // RR型
        L(root);
      }
    }
  }
}
```

AVL树的删除，书上没有提及。

## 并查集

并查集是一种维护集合的数据结构，主要针对合并、查找与集合三个单词。

并查集的实现就是一个数组，

```c
int father[N];
```

`father[i]`指的是元素`i`的父亲结点。在并查集中，一个集合只有一个结点满足`father[i]=i`，叫做根结点，把这个根结点看做是集合的标志。

并查集支持的操作，

初始化：

```c
for(int i =0;i<n;++i) {
  father[i] = i; // 初始化时，默认每个结点的父结点都是自身
}
```

查找当前集合的根结点：

```c
int findFather(int i) {
  while(father[i]!=i) {
    i = father[i];
  }
  return i;
}
```

合并操作，如果两个元素实际是属于同一集合，就合并这两个元素分属的两集合的根结点，只保留一个根结点：

```c
void Union(int x, int y) {
  int fatherX = findFather(x);
  int fatherY = findFather(y);
  if(fatehrX!=fatherY) {
    father[fatherX] = fatherY; // 集合X的根结点变换为集合Y的根结点
  }
}
```

路径压缩操作，是一种简化`father`数组，从而提高寻找根结点效率的方法，直接让`father[i]`指向根结点，而不是父结点。

```c
int findFather(int i) {
  int tmp = i;
  while(father[i]!=i) {
    i = father[i];
  }
  // 此时i是根结点，在此从i结点出发，逐次修改结点的父结点为根结点
  while(father[tmp]!=tmp) {
    int current_node = tmp;
    tmp = father[tmp];
    father[current_node] = i;
  }
  return i;
}
```

## 堆

堆是一课完全二叉树。它区别于BST和AVL的是，它只规定根结点要`>=`或者`<=`两个结点，根结点是当前树最大值的完全二叉树叫做大顶堆，反之叫做小顶堆。

如果现在我们已经有一棵完全二叉树，如何把它调整为一个大顶堆？

原则是从右到左，从下到上依次检查当前结点是否满足条件：

- 叶子结点无需调整
- 非叶子结点，将当前根结点与左、右子结点的最大值进行比较，如果当前根结点已经是最大值就无需变动；否则就互换，互换之后，由于左右子树在之前的调整过程中已经保证了大顶堆，那么新的根结点一定是最大值，只需要继续让被换以后的根结点与下一级子树进行比较即可。重复这个过程，直至无需再调整位置。

类似的，如果我们想新建一个小顶堆，只需要总是寻找根结点、左、右结点中的最小值即可，整体过程与上述过程类似。

从代码的角度分析，如何实现大顶堆？按照之前的讨论，完全二叉树可以使用根结点在位置$1$的数组进行存储。

```c
const int maxn = 110;
int heap[maxn]; // 堆
```

实现下上面的调整过程，让当前位置的$i$一直向下调整。

```c
// high一般为n
void downAdjust(int low, int high) {
  int i = low, child = 2 * i;
  while(child <= high) {
    if(child+1 <= high && heap[child] < heap[child]) {
      child = lchild + 1; // 寻找child中的最大值，此时为右结点
    }
    if(heap[i]>=heap[child]) 
      break; // 如果当前根结点已经保证是最大值，退出
    swap(heap[i], heap[child]);
    // 交换位置后，继续遍历子树
    i = child;
    child = 2 * child;
  }
}
```

建堆过程，从最右边，最低层的非叶子结点开始遍历，逐个执行上述函数：

```c
void createHeap() {
  // 如果完全二叉树有n个结点，那么非叶子结点有 取下限(n/2) 个
  for(int i = n/2; i>=1; i--) {
    downAdjust(i, n);
  }
}
```

删除堆顶元素，让最末端的叶子结点替换堆顶元素，然后执行向下调整

```c
void deleteTop() {
  heap[1] = heap[n--];
  downAdjust(1, n);
}
```

新增元素，让新元素放在堆的最末端，然后执行从下到上的调整

```c
// low一般为1， high是当前要进行向上调整的元素
void upAdjust(int low, int high) {
  int i = high, parent = high / 2;
  while(high>=low) {
    if (heap[parent]>=heap[i]) 
      break; // 父亲结点已经比当前结点大，无需再调整
    swap(heap[parent], heap[i]);
    i = parent;
    parent = parent / 2;
  }
}
void insert(int x) {
  heap[++n] = x;
  upAdjust(1, n);
}
```

堆的基本元素讲完后，我们可以使用堆来实现堆排序，下面是一个实现从小到大递增排序的例子：

```c
void heapSort() {
  // 建好堆之后，每次把堆顶元素和最末端的元素互换，然后执行向下调整
  crateHeap();
  for(int i = n; i>1; --i) {
    swap(heap[1], heap[i]);
    downAdjust(1, i-1);
  }
}
```

## 哈夫曼树

哈夫曼树的构建方法：

- 初始状态的n个结点看做是单独的n个二叉树
- 选择根结点值最小的两个结点进行合并，生成新的父结点，父结点的值是两个结点的和
- 重复上述过程，直至形成一棵完整的二叉树

叶子结点的权值$\times$根结点该叶子结点的路径长度，叫做改叶子结点的带权路径长度。一棵二叉树的带权路径长度Weight Path Length of tree（WPL）是所有叶子结点带权路径长度的和。哈夫曼树是满足一组叶子结点，带权路径长度最小的树。在建立哈夫曼树的过程中，能够保证哈夫曼树的WPL是最小的，但是可能存在多个哈夫曼树，比如可能同时存在多个相同的最小值结点。

哈夫曼树用来计算某个值的时候，可以考虑使用优先队列/小顶堆，每次取出最小的两个值合并后再入队。

现在考虑给字符用01编码的问题，如果随便给字符编码，就可能导致某个叶子结点的编码是另一个叶子结点编码的前缀，这样在使用这种编码过程的时候就可能出现问题。如果任一字符的编码都不是另一字符的编码前缀，这种编码方式叫做前缀编码。

对于这种情况，可以使用任意一棵二叉树，根结点到叶子结点的路径（左子树0，右子树1）作为该叶子结点的编码。它能够满足前缀编码的要求。

但是，另一个问题是如果我们希望一段字符串的01编码长度最小呢？这就可以使用哈夫曼编码来解决，统计字符串中各个字符的出现次数作为权值，然后建立哈夫曼树，就能够保证最后字符串的编码一定是最短的。
