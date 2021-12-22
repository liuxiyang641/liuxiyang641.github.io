---
title: 10-graph
notshow: false
date: 2021-11-29 21:45:44
categories:
- algorithm
tags:
- book
---

# 第10章 图算法专题

《算法笔记》第10章。

<!--more-->

## 图的基本定义

顶点（vertex）、边（edge）、出度、入度等不再赘述。

## 图的存储

这里从传统算法的角度讨论图的存储，两个基本办法：邻接矩阵和邻接表

```c
// 邻接矩阵
int maxv = 1000;
int G[maxv][maxv]; // 可使用1表示连通，适用于顶点数量较少，一般少于1000顶点的情况
// 邻接表
vector<int> G[n]; // n是顶点数量，每个数组元素是一个vector
// 如果要储存边权
struct Node {
  int v, dis;
}
vector<Node> G[n];
```

## 图的遍历

和前面在tree中讨论了很多次的一样，图的遍历同样是DFS和BFS，最大的区别在于

- 图不一定是连通的，可能存在多个连通分量，因此需要从每个顶点出发尝试遍历，并且不断记录已经访问过的顶点
- 遍历了所有顶点，不代表已经遍历了所有边。这一点需要特别注意

下面写出BFS和DFS的代码，以邻接表为例

```c
// DFS遍历grpah
void DFS(int u, int &stat_res) {
  vis[u] = ture; // 记当前顶点已访问
  for(int i = 0; i<G[u].size(); ++i) {
    if(vis[G[u][i]]==false) {
      // 如果新的下一级顶点还未访问，则DFS
      DFS(G[u][i], stat_res);
    }
  }
}
void DFSGraph() {
  for(int u = 0; u<n; ++u) {
    if(vis[u]==false) {
      int stat_res = 0; // 某些可能的统计数据
      DFS(u, stat_res);
    }
  }
}

// BFS遍历graph
void BFS(int u) {
  queue<int> q;
  vis[u] = ture;
  q.push(q);
  while(!q.empty()) {
    int u = q.top();
    q.pop();
    for(int i=0;i<G[u].size();++i) {
      if(vis[G[u][i]]==false)
      	q.push(G[u][i]);
      	vis[G[u][i]] = ture;
    }
  }
}
void BFSGraph() {
  for(int u = 0; u<n; ++u) {
    if(vis[u]==false) {
      BFS(u);
    }
  }
}
```

## 最短路径

### Dijkstra算法

求最短路径的经典算法，该算法能够从某个起点出发，寻找到其它所有顶点的最短路径。

基本思想：从还没有到达的剩余顶点中，选择一个最短距离的顶点，访问它；然后检查如果从这个新访问的顶点出发，看能否让剩余未到达的顶点的最短距离变小，如果可以就更新剩余顶点的最短距离；持续执行上一步，知道所有顶点都访问完毕。

实现时候的几个核心思路：

- 一个检查是否已经访问过的数组`bool vis[maxv]`，初始化时`false`
- 一个存储到不同顶点最短路径的数据`int d[maxv]`，初始化为`INF`，一个巨大的数字，可以是`e9`；结合`vis[maxv]`和`d[maxv]`就可以选出所有未到达顶点中具有最短路径的那个顶点

邻接矩阵版本的dijkstra算法：

```c
int INF = 1000000000;
// s是开始的起点编号
void Disjkstra(int s) {
  fill(d, d+n, INF);
  fill(vis, vis+n, false);
  d[s] = 0; // 开始顶点距离为0
  for(int i = 0; i<n; ++i) { // 开始访问n个顶点
    int u = -1;
    int MIN = INF;
    // 寻找还未访问的顶点中有最短路径的顶点
    for(int v=0; v<n; ++v) {
      if(vis[v]==false && d[v]<MIN) {
        u = v;
        MIN = d[v];
      }
    }
    if(u==-1) return; // 已经没有可以访问的顶点了，返回
    vis[u] = ture; // 访问节点u
    // 开始检查从u出发，能否让还未访问的顶点最短路径减小
    for(int v=0; v<n; ++v) {
      if(vis[v]==false && G[u][v]!=-1 && d[v]>d[u] + G[u][v]) {
        d[v] = d[u] + G[u][v]; // 更新最短路径
      }
    }
  }
}
```

上面的函数执行完毕后，`d[maxv]`中将保存所有最短路径距离。

接下来讨论，如何输出最短路径？

解决方法是记录每个顶点最短路径的前驱结点即可，开始结点的最短路径是自身

```c
int pre[maxv]; // 记录前驱
void Disjkstra(int s) {
  fill(d, d+n, INF);
  fill(vis, vis+n, false);
  d[s] = 0; // 开始顶点距离为0
  pre[s] = s; // 开始结点的前驱是自身
  for(int i = 0; i<n; ++i) { // 开始访问n个顶点
    int u = -1;
    int MIN = INF;
    // 寻找还未访问的顶点中有最短路径的顶点
    for(int v=0; v<n; ++v) {
      if(vis[v]==false && d[v]<MIN) {
        u = v;
        MIN = d[v];
      }
    }
    if(u==-1) return; // 已经没有可以访问的顶点了，返回
    vis[u] = ture; // 访问节点u
    // 开始检查从u出发，能否让还未访问的顶点最短路径减小
    for(int v=0; v<n; ++v) {
      if(vis[v]==false && G[u][v]!=-1 && d[v]>d[u] + G[u][v]) {
        d[v] = d[u] + G[u][v]; // 更新最短路径
        pre[v] = u; // 更新前驱
      }
    }
  }
}
```

然后通过递归就可以输出最短路径

```c
void DFSPath(int s, int u) {
  if(s==u) {
    printf("%d ", s);
    return;
  }
  DFSPath(s, pre[u]);
  printf("%d ", u);
}
```

当然在做题的时候，不会只有这么简单的要求，通常会有更多的要求，比如要求选择在最短路径中花费最少的一条，要求输出最短路径的数量等等。

下面是三种常见的应对策略：

- 给每条边新增边权，然后要求在多个最短路径中选择新增边权最好的情况

```c
// 新增的边权就类似于graph进行存储，同时用另一个新数组记录第二边权访问各个顶点时的情况
// 下面是核心代码
// 开始检查从u出发，能否让还未访问的顶点最短路径减小
for(int v=0; v<n; ++v) {
  if(vis[v]==false && G[u][v]!=-1) {
    if (d[v]>d[u] + G[u][v]) {
      d[v] = d[u] + G[u][v]; // 更新最短路径
      pre[v] = u;
    }
    else if (d[v] == d[u] + G[u]) {
      // 第二边权的更新，在最短路径不变的情况下选择最好的第二边权
      // 这里的cost代表路径的花费
      if(c[v] > cost[u][v] + c[u]) {
				c[v] = cost[u][v] + c[u];
        pre[v] = u;
      }
    }
  }
}
```

- 每个点新增了点权，要求在最短路径中，寻找点权最优的情况，类似于上面的方法

```c
for(int v=0; v<n; ++v) {
  if(vis[v]==false && G[u][v]!=-1) {
    if (d[v] > d[u] + G[u][v]) {
      d[v] = d[u] + G[u][v]; // 更新最短路径
      pre[v] = u;
      w[v] = w[u] + weight[v];
    }
    else if (d[v] == d[u] + G[u]) {
      // 点权的更新，在最短路径不变的情况下选择最好的点权
      // 这里希望点权w[v]越大越好
      if(w[v] < w[u] + weight[v]) {
				w[v] = w[u] + weight[v];
        pre[v] = u;
      }
    }
  }
}
```

- 求最短路径的数量，使用一个数组，记录最短路径数量即可

```c
for(int v=0; v<n; ++v) {
  if(vis[v]==false && G[u][v]!=-1) {
    if (d[v] > d[u] + G[u][v]) {
      d[v] = d[u] + G[u][v]; // 更新最短路径
      pre[v] = u;
      nums[v] = nums[u] // 到达顶点v的最短路径数量与达到顶点u一样
    }
    else if (d[v] == d[u] + G[u]) {
      // 说明此时从顶点u出发也可以最短路径的到达顶点v
      nums[v] += nums[u];
    }
  }
}
```

在上面的方法中，总是只保留最优的最短路径，这种情况不一定适用于所有的情形。下面介绍一种方法，总是先保留所有的最短路径，然后再从所有的最短路径中进行选择。

核心方法是，不再只保留一个前驱结点，而是保留所有的最短路径的前驱结点

```c
vector<int> pre[maxv];
```

新的方法

```c
void Disjkstra(int s) {
  fill(d, d+n, INF);
  fill(vis, vis+n, false);
  d[s] = 0; // 开始顶点距离为0
  pre[s].clear();
  pre[s].push_back(s); // 开始顶点的前驱是自身
  for(int i = 0; i<n; ++i) { // 开始访问n个顶点
    int u = -1;
    int MIN = INF;
    // 寻找还未访问的顶点中有最短路径的顶点
    for(int v=0; v<n; ++v) {
      if(vis[v]==false && d[v]<MIN) {
        u = v;
        MIN = d[v];
      }
    }
    if(u==-1) return; // 已经没有可以访问的顶点了，返回
    vis[u] = ture; // 访问节点u
    // 开始检查从u出发，能否让还未访问的顶点最短路径减小
    for(int v=0; v<n; ++v) {
      if(vis[v]==false && G[u][v]!=-1) {
        if(d[v]>d[u] + G[u][v]) {
          d[v] = d[u] + G[u][v]; // 更新最短路径
          // 更新前驱
          pre[v].clear();
          pre[v].push_back(u);
        } else if(d[v]>d[u] + G[u][v]) {
          pre[v].push_back(u); // 记录新的可能的最短路径的前驱
        }
      }
    }
  }
}
```

遍历所有的最短路径：

```c
void DFSPath(int s, int u, vector<int> &tmpPath, int &optValue, vector<int> &optPath) {
  if(u==s) {
    tmpPath.push_back(u);
    int value;
    if(当前最短路径的value优于optvalue) {
      optValue = value;
      optPath = tmpPath;
    }
    tmpPath.pop_back();
    return;
  }
  tmpPath.push_back(u); // 当前路径加入新的结点
  // 开始遍历所有的前驱结点
  for(int i =0;i<pre[u].size();++i) {
    DFSPath(s, pre[i], tmpPath, optValue, optPath);
  }
  // 退出当前结点，返回上一级
  tmpPath.pop_back();
}
```

### Bellman-Ford算法和SPFA算法

在dijkstra算法中，如果遇到图有负权边，由于该算法会直接选择该负边，而忽略了其它可能的路径，可能造成某些通过非负权边可以访问到的顶点没有被访问到。它无法较好的处理负权边。

对于以上问题，同样是针对单源最短路径问题，有bellman-ford算法，以及其改进版本SPFA算法可以解决。

Bellman-ford算法的基本思想：

- 对图中的每个边进行`V-1`轮的检查。在每一轮的检查中，如果发现通过边`[u][v]`，可以让顶点`v`的最短路径缩短，就进行替换，这一点类似于Dijkstra算法，区别在于Bellman算法是遍历每条边，保证所有的边都会参与判定过程。进行`V-1`轮检查的原因是，某个结点到开始顶点的最短路径长度不会超过`V`（包括开始顶点），如果不考虑都有的开始顶点，只需要最多`V-1`步就可以到达任意连通的结点。
- 之后，再进行一轮检查，如果发现还有某个边，可以更新当前的最短路径，可以判定图中存在源点可达的负环（也就是循环一轮后，发现总的边权减少了）。请注意，这样无法判定图中是否有源点不可达的负环。

以邻接表为例的代码：

```c
// 返回ture表示无源点s可达的负环
bool bellman(int s) {
  d[s] = 0;
  // 开始n-1轮检查
  for(int i = 0; i < n-1; ++i) {
    // 开始遍历所有边
    for(int u = 0; u < n; ++u) {
      for(int j = 0; j < Adj[u].size(); ++j) {
        int v = Adj[u][j].v;
        int dis = Adj[u][j].dis;
        if(d[v] > d[u] + dis) {
          d[v] = d[u] + dis;
        }
      }
    }
  }
  // 开始判断是否有源点可达的负环
  for(int u = 0; u < n; ++u) {
      for(int j = 0; j < Adj[u].size(); ++j) {
        int v = Adj[u][j].v;
        int dis = Adj[u][j].dis;
        if(d[v] > d[u] + dis) {
          return false;
        }
      }
   }
	return ture;
}
```

上述算法每次要遍历所有的边，实际上只有最短路径`d`发生变化的顶点出发的边才需要进行判断，因此可以使用一个队列存储所有最短路径发生变化的顶点，出队后，再把发生最短路径变化且不再队列中的顶点入队。如果发现队空了，可以判断没有可达的负环；如果有某个顶点入队次数超过了`V`（也就是最短路径发生变化超过了`V`次），可以判断存在可达的负环。

经过上述改进后的算法就叫做SPFA算法（Shortest Path Faster Algorithm），该算法在大多数的图中都非常高效。

```c
// 返回ture表示无源点s可达的负环
bool SPFA(int s) {
  queue<int> q;
  q.push(s);
  bool inqueue[n]={false};
  int inqueueNum[n]={0};
  inqueue[s] = true;
  d[s] = 0;
  inqueueNum[s] = 1;
  
  while(!q.empty()) {
    int u = q.top();
    q.pop();
    for(int j = 0; j < Adj[u].size(); ++j) {
        int v = Adj[u][j].v;
        int dis = Adj[u][j].dis;
        if(d[v] > d[u] + dis) {
          d[v] = d[u] + dis;
          // 顶点v的最短路径发生变化
          if(inqueue[v]==false) {
            q.push(v);
            inqueue[v] = true;
            inqueueNum[v]++;
            if(inqueueNum[v]>=n) // 入队次数超过或者达到了n
              return false; 
          }
        }
      }
  }
  return ture;
}
```

### Floyd算法

Floyd可以解决全源最短路径的问题，也就是说询问任意两个点之间的最短距离，该问题就限制了问题可以查询的顶点数量在200以内，所以总是可以使用邻接矩阵的方法解决。核心思想是，如果顶点`k`为中介时，可以使得顶点`i`到顶点`j`的距离缩短，就使用顶点`k`为中介。

代码：

```c
void Floyd(int s) {
  for(int k =0; k<n; ++k) {
    // 开始遍历所有顶点组合
    for(int i =0; i<n; ++i) {
      for(int j=0; j<n; ++j) {
        if(dis[i][k]!=INF && dis[k][j]!=INF && dis[i][k]+dis[k][j]<dis[i][j]) {
          dis[i][j] = dis[i][k]+dis[k][j];
        }
      }
    }
  }
}
```

## 最小生成树

最小生成树是从一个无向图当中，获取一课树，这棵树满足

- 包括了所有的图顶点
- 所有的边都是图中原有的边
- 该树的边权和最小

由于是一棵树，所以最小生成树一定有`V-1`条边。最小生成树的根结点可以是任意的结点（试想下一棵树，如果没有特殊的性质，我们当然可以把任意一个数结点当做是根结点，然后重新排列成树的层级形状）。当然在题目中，一般会指定要从哪个结点出发生成最小生成树。

### Prime算法

prime算法和dijkstra算法很相似，区别在于prime选择下一步访问的图顶点时不是考虑到起源结点最短距离，而是到整个已访问结点集合的最短距离（具体的说，访问新顶点，检查下新访问顶点到未访问顶点的距离，看能否让距离减小，不需要考虑之前新访问顶点的最短距离）。

下面写一下邻接表版本的prime算法：

```c
int INF = 1000000000;
bool vis[maxn];
int dis[maxn];
int prime(int s) {
  fill(vis, vis+n, false);
  fill(dis, dis+n, INF);
  dis[s] = 0; // 起源顶点的最短距离设置为1
  int ans = 0; // 记录生成树的边权和
  // 开始访问所有顶点，总共访问n次
  for(int i = 0; i < n; ++i) {
    int MINDIS = INF;
    int u = -1;
    // 下面这段代码可以使用小顶堆或者优先队列维护，就无须总是遍历所有的顶点了
    for(int v = 0; v<n; ++v) {
			if(vis[v]==false && dis[v]<MINDIS) {
        // 寻找当前最短距离最小的顶点
        MINDIS = dis[v];
        u = v;
      }
    }
    if(u==-1) return ans;
    vis[u] = true; // 访问顶点u
    ans += dis[u]; // 累积边权和
    // 开始更新未访问顶点的最短距离
    // 检查顶点u的相连顶点
    for(int j = 0; j<Adj[u].size(); ++j) {
      int v = Adj[u][j].v;
      int dis_uv = Adj[u][j].dis;
      // 如果顶点v未访问，并且距离顶点u的边权更小
      // 这一行是prime区别于dijkstra的核心，不考虑之前顶点u的最短距离
			if(vis[v] == false && dis[v] > dis_uv) {
        dis[v] = dis_uv;
      }
    }
  }
  return ans;
}
```

### kruskal算法

Kruskal算法（克鲁斯卡尔算法）的思想很简单，使用边贪心的思路：

- 按照边权从小到大排序所有边
- 认为所有图顶点一开始是独立不连通的块（并查集的初始状态）
- 遍历所有排好序的边，如果一条边的两个顶点不在同一个连通块（并查集寻找集合根结点），就加入这个边，连通两个连通块（并查集合并）；如果两个顶点已经处于同一个连通块，就略过该边
- 重复上一步直至所有边遍历完毕或者已经选择了`V-1`个边

代码示意：

```c
struct edge {
  int u, v;
  int dis;
} E [maxe];

bool cmp(edge e1, edge e2) {
  return e1.dis < e2.dis;
}

int father[maxn]; // 记录顶点所属的连通块/集合

int findFather(int i) {
  int tmp_i = i;
  while(father[i]!=i) {
    i = father[i];
  }
  // 压缩并查集路径
  while(i!=father[tmp_i]) {
    int tmp_z = tmp_i;
    tmp_i=father[tmp_i];
    father[tmp_z] = i; // 直接指向集合的根结点
  }
  return i;
}

int kruskal() {
  int ans = 0;
  for(int i = 0; i<n; ++i) {
		father[i] = i; // n个不连通块
  }
  sort(E, E+edge_num, cmp);
  int u, v;
  int tree_count = 0;
  for(int i=0; i<edge_num; ++i) {
    u = E[i].u;
    fatherU = findFather(u);
    v = E[i].v;
    fatherV = findFather(v);
    dis = E[i].dis;
	  if(fatherU!=fatherV) {
      father[fatherU] = father[fatherV]; // 合并两个并查集，根结点合并
      ans += dis; // 边加入生成树
      tree_count++;
      if(tree_count==n-1) break; // 如果已经找到足够的生成树边
    }
  }
  if(tree_count!=n-1) return -1; // 有顶点无法连通
  return ans;
}
```

## 拓扑排序

一个有向图的任意顶点都不可能通过一些有向边返回，这样的有向图叫做有向无环图。

检查一个有向无环图的办法可以通过检查图的拓扑排序能否包括所有的图顶点。拓扑排序是指如果在图中存在`u->v`，则`u`在拓扑排序中一定在`v`前，`u`是`v`的先导元素。

解决思路是，使用一个队列存储所有入度为0的顶点，出队队首元素，访问该顶点，然后删除所有以该顶点为起点的边，如果有顶点入度变为了0，就入队。重复上述过程直到队列为空。检查此时访问的元素，如果存在部分顶点为访问，则说明有向图中存在环。

邻接表版本的代码：

```c
int inDegree[maxn]; // 记录所有顶点的入度
bool topologicalSort() {
  queue<int> q;
  // 所有入度是0的顶点入队
  for(int i=0;i<n;i++) {
    if(inDegree[i]==0) {
      q.push(i);
    }
  }
  int zeroDegreeCount = 0;
  while(!q.empty()) {
    int u = q.front();
    q.pop();
    zeroDegreeCount++;
    printf("%d ", u); // 访问队顶元素，输出拓扑排序
    for(int j = 0; j < Adj[u].size(); ++j) {
      int v = Adj[u][j].v;
      inDegree[v]--;
      if(inDegree[v]==0) {
        q.push(v);
      }
    }
  }
  if(zeroDegreeCount==n) return true; // 返回true，没有环
  return false; // 有环
}
```

## 关键路径

AOV（Activity On Vertex）网络是顶点表示活动，顶点可以有点权，边没有边权，代表优先级。上一节的拓扑排序就是用来寻找AOV网络上的一种活动排序。

AOE（Activity On Edge）网络是边表示活动，顶点表示事件，顶点没有点权，边有边权。AOE通常用在工程场景中，边表示从一个事件到另一事件需要的时间/代价等。

AOV和AOE网络的边都表示某种优先级，因此不会存在环，都是有向无环图。

AOV网络总是可以转换为AOE网络，试想下只需要把AOV中的顶点拆分为两个顶点作为事件开始与结束，这两个顶点中的有向边边权就是原顶点的点权，剩下的AOV原有边边权设置为0。

AOE网络总是可以通过添加额外的起点和汇点，形成只有一个起点，一个汇点的图。

关键路径：对于AOE网络中的最长路径，叫做关键路径；关键路径上的所有活动叫做关键活动；关键路径表示要完成AOE网络中的所有活动所需要的最少时间，关键活动是无法拖延完成的活动。



关键路径的寻找方法，核心在于寻找顶点`i`（事件）的最早开始时间和最晚开始时间，最早开始时间是从起点开始就马不停蹄的完成所有事件，只要顶点`i`的所有先导顶点完成了，就立刻开始完成顶点`i`。顶点`i`的最晚开始时间，是从终点开始反向计算，只要不延误后续顶点的最晚开始时间就可以。



实现的时候，对于每个顶点，维护数组`ve[maxn]`保存顶点的最早开始时间；数组`vl[maxn]`保存顶点的最迟开始时间。计算好这两个数组之后，遍历所有的边`u->v`，计算`u->v`的最早开始时间`ve[u]`和最晚开始时间`vl[v]-dis[u->v]`。

按照拓扑排序，可以计算出各个顶点的最早开始时间`max`（所有先导顶点的最早时间+先导边时间）；然后按照拓扑排序的反向顺序，计算各个顶点的最晚开始时间`min`（所有后续顶点的最晚开始时间-后续边时间）。

代码实现：

```c
stack<int> topoloStack; // 保存拓扑排序序列
int ve[maxn];
int vl[maxn];

int inDegree[maxn];
bool topologicalSort() {
  fill(ve, ve+n, 0);
  queue<int> q;
  // 所有入度是0的顶点入队
  for(int i=0;i<n;i++) {
    if(inDegree[i]==0) {
      q.push(i);
    }
  }
  int zeroDegreeCount = 0;
  while(!q.empty()) {
    int u = q.front();
    q.pop();
    zeroDegreeCount++;
    topoloStack.push(u);  // 加入拓扑排序
    // 对于顶点u的所有后续结点，顶点u都是先导顶点
    for(int j = 0; j < Adj[u].size(); ++j) {
      int v = Adj[u][j].v;
      int dis = Adj[u][j].dis;
      inDegree[v]--;
      if(inDegree[v]==0) {
        q.push(v);
      }
      // 如果以顶点u为先导，完成活动后到达顶点v的最早开始时间更长
      if(ve[v] < ve[u] + dis) {
        ve[v] = ve[u] + dis;
      }
    }
  }
  if(zeroDegreeCount==n) return true; // 返回true，没有环
  return false; // 有环
}

int criticalPath() {
  if(!topologicalSort()) return -1; // 有环
  // 开始计算vl[n]
  fill(vl, vl+n, ve[n-1]); // 初始化vl值为汇点的最晚开始时间（等于汇点的最早开始时间），也就是关键路径长度
  while(!topoloStack.empty()) {
    int u = topoloStack.top();
    topoloStack.pop();
    // 对于顶点u的所有后续结点
    for(int j = 0; j < Adj[u].size(); ++j) {
      int v = Adj[u][j].v;
      int dis = Adj[u][j].dis;
      // 如果到达顶点v的最晚开始时间更小
      if(vl[u] > vl[v] - dis) {
        vl[u] = vl[v] - dis;
      }
    }
  }
  // 开始遍历所有边，寻找关键活动，也就是不能延误开始的边
  for(int u = 0; u < n; ++u) {
    for(int j = 0; j < Adj[u].size(); ++j) {
      int v = Adj[u][j].v;
      int dis = Adj[u][j].dis;
      int e_edge = ve[u]; // 边的最早开始时间
      int l_edge = vl[v] - dis; // 边的最迟开始时间
      if(e_dege == l_edge) {
				printf("%d->%d\n", u, v);
      }
    }
  }
  return ve[n-1];
}
```

