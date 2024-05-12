# 一. **markdown**
## 什么是Markdown？
Markdown是一种轻量级的标记语言，以使用简单的文本格式来书写和编辑文档，同时具有一定的格式化效果。通过使用Markdown语法，用户可以快速地将纯文本转换为格式化的文档，例如标题、列表、链接、图片等。Markdown语法简单直观，易于学习和使用，逐渐成为了各种平台上书写文档的常用格式，甚至还可以实现Markdown文档对PDF、Word等格式的快速转换。
Markdown的文件通常以.md或.markdown为扩展名。
![图片](https://image.itbaima.cn/markdown/2024/04/01/vdfIcRaSolpG1Eq.png)
可以创建一个TXT文本文档，然后将其后缀改为.md即可开始编辑此Markdown文档。
***
# 二. **git与github**
##  1. git安装与配置
`git config --global user.name "Your Name"`
`git config --global user.email "email@example.com"`

`git config --list`

`git init`
工作区有一个隐藏目录.git，Git的版本库。最重要的就是称为stage（或者叫index）的暂存区，还有Git为我们自动创建的第一个分支master，以及指向master的一个指针叫HEAD。
`git add.`把项目中所有修改文件添加进去，实际上就是把文件修改添加到暂存区；
`git commit`提交更改，实际上就是把暂存区的所有内容提交到当前分支。
`git status`查看当前工作目录的状态
`git diff`查看修改内容

* 穿梭前，用`git log`可以查看提交历史，以便确定要回退到哪个版本。
* 要重返未来，用`git reflog`查看命令历史，以便确定要回到未来的哪个版本。

* 一种是 文件 自修改后还没有被放到暂存区，现在，撤销修改就回到和版本库一模一样的状态；

  `git checkout -- readme.txt (其实是从版本库重新检出一份文件)`
* 一种是readme.txt修改后已经添加到暂存区后，执行以下两条命令

  `git reset HEAD readme.txt `
  `git checkout -- readme.txt`

* 忽略文件
不需提交到本地仓库的文件可以忽略
把不需要提交的文件添加到.gitignore，就不再提交git仓库
***
# 三. **cindrad**
## 1. correct
### I. __init__.py
__init__.py是Python中用于标识一个包（package）的初始化文件。当Python解释器在导入一个包的时，会首先查找并执行该包下的__init__.py文件。__init__.py文件可以为空，也可以包含一些初始化代码，用于定义包的属性、变量、函数等。通常情况下，__init__.py文件用于执行一些初始化操作，例如导入模块、设置包的属性或变量等。
在Python 3.3及以后的版本中，__init__.py文件不再是必需的，Python引入了隐式命名空间包（Implicit Namespace Packages）的概念，允许在没有__init__.py文件的情况下定义包。但是为了向后兼容性和更好的包结构管理，建议在包目录下始终保留一个__init__.py文件。

`from .dealias import dealias`

意思是当前目录下dealias 的模块中导入名为 "dealias" 的函数。这行代码通常用于在当前模块中使用 "dealias" 函数，而不必在代码中重复定义该函数。
### II._unwrap_2d.pyx
是一个Cython源代码文件，用于实现二维相位解包算法。该算法通常用于处理相位不连续或不连贯的数据，以便更好地理解和分析这些数据。通过使用Cython编写该算法，可以提高其性能和效率，使其更适合处理大规模的数据集。
* 这段代码是一个 Cython 函数的定义，用于执行 2D 相位反包裹操作。具体来说，它定义了一个名为 "unwrap_2d" 的函数，该函数接受三个参数：一个包含双精度浮点数的二维数组 "image"，一个包含无符号字符的二维数组 "mask"，以及一个用于指示是否在 x 和 y
方向上进行包裹的布尔值数组"wrap_around"。在函数内部，它调用了名为 "unwrap2D" 的外部 C 函数，该函数在 "unwrap_2d_ljmu.c" 文件中定义。这个外部函数接受指向包裹图像、未包裹图像、输入掩码以及图像宽度、高度、x 和 y 方向的包裹标志的指针，并执行相应的 2D 相位反包裹操作。总的来说，这段代码实现了一个用于 2D 相位反包裹的函数，并通过 Cython 来调用外部 C 函数来执行实际的操作。

`cdef extern from "unwrap_2d_ljmu.c":`
*  表示从名为`unwrap_2d_ljmu.c`的C语言源文件中导入函数或变量的声明。这样可以在Cython代码中调用C语言代码中定义的函数或变量，实现Cython与C语言之间的交互和互操作。这种方法通常用于优化性能或利用现有的C语言库来完成一些特定的任务。

` void unwrap2D(double* wrapped_image,double* unwrapped_image,unsigned char* input_mask,
int image_width, int image_height,int wrap_around_x, int wrap_around_y)`

* 这是一个C语言函数的声明，该函数名为`unwrap2D`，接受以下参数：参数的作用是对输入的包裹相位图像进行解包处理，并将结果存储在`unwrapped_image`中。输入掩膜`input_mask`用于指示哪些像素需要进行解包操作。`image_width和`image_height`表示图像的尺寸，`wrap_around_x`和`wrap_around_y`表示是否在X轴和Y轴上进行环绕处理。

|            **参数**            |           **功能**           |
|:----------------------------:|:----------------------:|
|   `double* wrapped_image`    |     指向包裹相位图像数据的指针      | 
|  `double* unwrapped_image`   |     指向解包相位图像数据的指针      | 
| `unsigned char* input_mask`  |      指向输入掩膜数据的指针       | 
|      `int image_width`       |         图像的宽度          | 
|      `int image_height`      |         图像的高度          | 
|     `int wrap_around_x`      | X轴是否环绕（wrap around）的标志 | 
|     `int wrap_around_y`      | Y轴是否环绕（wrap around）的标志 | 




