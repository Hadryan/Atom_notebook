# ___2019 - 12 - 24 Go Basic___
***
# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2019 - 12 - 24 Go Basic___](#2019-12-24-go-basic)
  - [目录](#目录)
  - [Install](#install)
  	- [Install go](#install-go)
  	- [Install gophernotes](#install-gophernotes)
  	- [Install lgo](#install-lgo)
  - [基础语法](#基础语法)
  	- [foo](#foo)
  	- [Go 程序的一般结构](#go-程序的一般结构)
  	- [Go 数据类型](#go-数据类型)
  	- [for 循环语句](#for-循环语句)
  	- [if 条件语句](#if-条件语句)
  	- [switch 语句](#switch-语句)
  	- [func 函数](#func-函数)
  	- [defer 延迟调用](#defer-延迟调用)
  - [数据结构](#数据结构)
  	- [指针](#指针)
  	- [struct 结构体](#struct-结构体)
  	- [数组](#数组)
  	- [slice 切片](#slice-切片)
  	- [range 迭代遍历](#range-迭代遍历)
  	- [map 字典](#map-字典)
  - [方法和接口](#方法和接口)
  	- [方法](#方法)
  	- [接口](#接口)
  	- [Stringers 接口](#stringers-接口)
  	- [Error 错误接口](#error-错误接口)
  	- [Readers 接口](#readers-接口)
  	- [Web 服务器](#web-服务器)
  	- [Image 图片接口](#image-图片接口)
  - [并发](#并发)
  	- [goroutine](#goroutine)
  	- [channel](#channel)
  	- [select](#select)

  <!-- /TOC -->
***

# Install
## Install go
  - [Go 指南](http://go-tour-zh.appspot.com/flowcontrol/4)
  - [Getting Started install go](https://golang.org/doc/install)
  - **hello world**
    ```go
    package main

    import "fmt"

    func main() {
      fmt.Printf("hello, world\n")
    }
    ```
    ```sh
    go run hello.go
    # Or
    go build hello.go
    ./hello
    # hello, world
    ```
## Install gophernotes
  - [Jupyter kernels](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels)
  - [gophernotes - Use Go in Jupyter notebooks and interact](https://github.com/gopherdata/gophernotes)
    ```sh
    jupyter-notebook --> New --> Go
    ```
  - **Q / A**
    ```sh
    ''' Q
    Package libzmq was not found in the pkg-config search path
    '''
    ''' A
    Add libzmq.pc path to PKG_CONFIG_PATH env
    '''
    locate libzmq
    # /opt/anaconda3/lib/libzmq.so.5.2.1
    # /opt/anaconda3/lib/pkgconfig/libzmq.pc
    # ...
    # /usr/lib/x86_64-linux-gnu/libzmq.so.5
    # /usr/lib/x86_64-linux-gnu/libzmq.so.5.1.5

    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/anaconda3/lib/pkgconfig
    ```
    ```sh
    ''' Q
    zmq4 was installed with ZeroMQ version 4.3.1, but the application links with version 4.2.5
    '''
    ''' A
    libzmq.so version in /usr/lib/x86_64-linux-gnu/ is not compatible with libzmq.pc
    '''
    ls -l /opt/anaconda3/lib/libzmq.so.5*
    # lrwxrwxrwx 1 leondgarse leondgarse      15 十月 12 10:07 /opt/anaconda3/lib/libzmq.so.5 -> libzmq.so.5.2.1
    # -rwxrwxr-x 2 leondgarse leondgarse  731928 二月  1  2019 /opt/anaconda3/lib/libzmq.so.5.2.1

    ls -l /usr/lib/x86_64-linux-gnu/libzmq.so.5*
    # lrwxrwxrwx 1 root root     15 七月  3 22:30 /usr/lib/x86_64-linux-gnu/libzmq.so.5 -> libzmq.so.5.1.5
    # -rw-r--r-- 1 root root 630464 七月  3 22:30 /usr/lib/x86_64-linux-gnu/libzmq.so.5.1.5

    sudo rm /usr/lib/x86_64-linux-gnu/libzmq.so.5
    sudo ln -s /opt/anaconda3/lib/libzmq.so.5.2.1 /usr/lib/x86_64-linux-gnu/libzmq.so.5
    ls -l /usr/lib/x86_64-linux-gnu/libzmq.so.5*
    # lrwxrwxrwx 1 root root     34 十二 25 09:29 /usr/lib/x86_64-linux-gnu/libzmq.so.5 -> /opt/anaconda3/lib/libzmq.so.5.2.1
    # -rw-r--r-- 1 root root 630464 七月  3 22:30 /usr/lib/x86_64-linux-gnu/libzmq.so.5.1.5
    ```
  - **hello world**
    ```go
    import "fmt"
    fmt.Printf("hello, world\n")
    ```
    ![](images/jupyter_go.png)
## Install lgo
  - [yunabe/lgo Go (golang) Jupyter Notebook kernel and an interactive REPL](https://github.com/yunabe/lgo)
    ```sh
    jupyter-notebook --> New --> Go(lgo)
    ```
  - **Jupyter Console**
    ```sh
    alias igo='jupyter console --kernel lgo'
    igo
    ```
    ```go
    In [1]: import "fmt"

    In [2]: fmt.Printf("Hello world\n")
    Hello world
    12
    <nil>
    ```
***

# 基础语法
## foo
  - **字符串连接** Go 语言的字符串可以通过 + 实现
    ```go
    package main
    import "fmt"
    func main() {
        fmt.Println("Google" + "Runoob")
    }
    ```
  当两个或多个连续的函数命名参数是同一类型，则除了最后一个类型之外，其他都可以省略
  命名返回值
  Go 的返回值可以被命名，并且像变量那样使用。

  返回值的名称应当具有一定的意义，可以作为文档使用。

  没有参数的 return 语句返回结果的当前值。也就是`直接`返回。

  直接返回语句仅应当用在像下面这样的短函数中。在长的函数中它们会影响代码的可读性。
  ```go
  package main
  import "fmt"

  func split(sum int) (x, y int) {
      x = sum * 4 / 9
      y = sum - x
      return
  }

  func main() {
      fmt.Println(split(17))
  }
  ```
  - 图片
    ```go
    package main

    import "code.google.com/p/go-tour/pic"

    func Pic(dx, dy int) [][]uint8 {
    }

    func main() {
    	pic.Show(Pic)
    }
    ```
## Go 程序的一般结构
  - **Go 程序文件夹结构**
    - 从指定文件夹下导入时，会导入所有的 go 文件
    - 要求该文件夹下的所有 go 文件有统一的包名，包名最好跟文件名相同，避免歧义
    - 包外调用方法名首字母必须为大写
    ```sh
    $ tree
    .
    ├── foo
    │   ├── add.go
    │   └── sub.go
    └── test.go
    ```
    **文件内容**
    ```go
    // $ cat test.go
    package main
    import "fmt"
    import "./foo"

    func main() {
        fmt.Println("Hello World!", goo.Add(1, 1), goo.Sub(1, 1))
    }

    // $ cat foo/add.go
    package goo
    func Add(x, y int) int {
        return x + y
    }

    // $ cat foo/sub.go
    package goo
    func Sub(x,y int) int {
        return x - y
    }
    ```
    **运行**
    ```sh
    go run test.go
    # Hello World! 2 0
    ```
## Go 数据类型
  - Go语言中，使用 **大小写** 来决定该常量、变量、类型、接口、结构或函数是否可以被外部包所调用，即 private / public
  - Go 中的字符串只能使用 **双引号**
    ```go
    aa := "aa"
    fmt.Printf("%T", aa)
    // string6
    ```
  - **包 package / import** Go 程序是通过 package 来组织的
    - 只有 **package** 名称为 **main** 的包可以包含 main 函数，一个可执行程序有且仅有一个 main 包
    - 通过 **import** 关键字来导入其他非 main 包，使用 `<PackageName>.<FunctionName>` 调用
    - 文件名 / 文件夹名与包名没有直接关系，不需要一致，但按照惯例，最好时一致，同一个文件夹下的文件只能有一个包名，否则编译报错
    - 可以使用 **()** 打包导入多个
    ```go
    package main  // 当前程序的包名
    import . "fmt"  // 导入其他包
    import (  // 同时导入多个
        "fmt"
        "math/rand"
    )
    import fmt2 "fmt" // package 别名
    import . "fmt"  // 表示省略调用，调用该模块里面的函数可以不写模块名
    ```
  - **数据定义 const / var / type**
    - Go 语言的 **类型** 在 **变量名之后**
    - `var` 语句可以定义在包或函数级别，即在函数外或函数内
    - 变量在没有初始化时默认为 `0` 值，数值类型为 `0`，布尔类型为 `false`，字符串为 `""` 空字符串
    - **const** 关键字定义常量
    - **var** 关键字定义变量，在函数体外部使用则定义的是全局变量
    - **type** 关键字定义结构 struct 和接口 interface
    ```go
    const PI = 3.14 // 常量
    var name = "gopher" // 变量的声明和赋值
    var c, python, java bool
    var c, python, java = true, false, "no!"

    type newType int  // 一般类型声明
    type gopher struct{}  // 结构
    type golang interface{} // 接口
    ```
  - **短声明变量 `:=`** 可以用于替代 `var` 定义，只能用于函数内，变量的类型由右值推导得出，常量不能使用该语法定义
    ```go
    k := 3
    c, python, java := true, false, "no!"
    ```
  - **基本类型**
    ```go
    bool
    string
    int  int8  int16  int32  int64
    uint uint8 uint16 uint32 uint64 uintptr
    byte // uint8 的别名
    rune // int32 的别名，代表一个Unicode码
    float32 float64
    complex64 complex128
    ```
    ```go
    import (
        "fmt"
        "math/cmplx"
    )

    var (
        ToBe   bool       = false
        MaxInt uint64     = 1<<64 - 1
        z      complex128 = cmplx.Sqrt(-5 + 12i)
    )

    fmt.Printf("z is of type %T\n", z)
    // z is of type complex128
    ```go
    ```
    const (
        Middle = 1.0
        Big   = Middle << 100
        Small = Middle >> 99

    )
    fmt.Printf("Middle: %T, Small: %T\n", Middle, Small)
    // Middle: float64, Small: int
    fmt.Printf("Big: %T\n", Big)
    // 1:25: Big (untyped int constant 1267650600228229401496703205376) overflows int
    fmt.Printf("Big: %e, %T\n", Big * 1.0, Big * 1.0)
    // Big: 1.267651e+30, float64
    ```
  - **类型转换** Go 在不同类型之间的项目赋值时，需要 **显式转换**
    ```go
    var i int = 42
    var f float64 = float64(i)
    var u uint = uint(f)
    ```
    ```go
    i := 42
    f := float64(i)
    u := uint(f)
    ```
## for 循环语句
  - **for 循环** Go 只有一种循环结构
    - 基本的 for 循环体使用 `{}`，循环条件 **不使用** `()`
    - 循环条件之间使用 **;** 分割，分别为 `前置语句，初始值` / `结束条件` / `后置语句，值调整`
    - 前置语句 / 后置语句可以为空，此时可以省略 `;`，类似于 **while**
    - **死循环** 如果省略了循环条件，循环就不会结束，`for { ... }`
    ```go
    sum := 0
    for i := 0; i < 10; i++ {
        sum += i
    }
    fmt.Println(sum)
    ```
    ```go
    for ; sum < 1000; {
        sum += sum
    }
    ```
    ```go
    for sum < 1000 {
        sum += sum
    }
    ```
## if 条件语句
  - **if 条件** 判断条件 **不使用** `()`，执行语句使用 `{}`
    ```go
    import "math"
    x := -2.0
    result := ""
    if x < 0 {
        result = fmt.Sprint(math.Sqrt(-x)) + "i"
    }
    ```
  - `if` 语句可以在条件之前执行一个简单的语句，由这个语句定义的变量的作用域仅在 **if 以及 if 对应的 else 范围** 之内
    ```go
    x := 4.0
    lim := 10.0
    if v := math.Pow(x, 2); v < lim {
        fmt.Println(v)
    } else {
        fmt.Printf("%g >= %g\n", v, lim)
    }
    // 16 >= 10
    ```
## switch 语句
  - **switch** 从上到下判断条件，当匹配成功时停止，`fallthrough` 使分支 `case` 继续向下执行，否则终止
    ```go
    import "runtime"
    fmt.Print("Go runs on ")

    switch os := runtime.GOOS; os {
    case "darwin":
        fmt.Println("OS X.")
    case "linux":
        fmt.Println("Linux.")
    default:
        // freebsd, openbsd,
        // plan9, windows...
        fmt.Printf("%s.", os)
    }
    ```
  - **没有条件的 switch** 可以用更清晰的形式编写长的 if-else 链
    ```go
    import "time"
    t := time.Now()
    switch {
        case t.Hour() < 12:
            fmt.Println("Good morning!")
        case t.Hour() < 17:
            fmt.Println("Good afternoon.")
        default:
            fmt.Println("Good evening.")
    }
    ```
## func 函数
  - **函数声明 func**，函数可以没有参数或接受多个参数，类似于变量定义，返回值类型在函数名之后
    ```go
    func main(argc int, argv []string) int

    // 由 main 函数作为程序入口点启动
    func main() { // { 不能单独一行
        Println("Hello World!")
    }

    // 函数名首字母大写为 public
    func Add(x int, y int) int {
        return x + y
    }

    // 函数名首字母小写为 private
    func swap(x, y string) (string, string) {
        return y, x
    }
    ```
  - **函数值** 函数也可以作为一个值
    ```go
    import "math"
    hypot := func(x, y float64) float64 {
        return math.Sqrt(x * x + y * y)
    }

    fmt.Println(hypot(3, 4))
    // 5
    ```
  - **函数的闭包** 函数的返回值是另一个函数
    ```go
    func adder() func(int) int {
        sum := 0
        return func(x int) int {
            sum += x
            return sum
        }
    }

    pos, neg := adder(), adder()
    for i := 0; i < 5; i++ {
        fmt.Println(pos(i), neg(-2*i))
    }
    // 0 0
    // 1 -2
    // 3 -6
    // 6 -12
    // 10 -20
    ```
## defer 延迟调用
  - **defer 语句** 延迟函数的执行在上层函数返回之后，延迟调用的参数会立刻生成，但是在上层函数返回前函数都不会被调用，可以用于释放资源等
    ```go
    i := 0
    defer fmt.Println(i)
    i++
    // 0
    ```
  - **defer 栈** 延迟的函数调用被压入一个栈中，当函数返回时，会按照后进先出的顺序调用被延迟的函数调用
    ```go
    fmt.Println("counting")

    for i := 0; i < 10; i++ {
        defer fmt.Print(i)
    }

    fmt.Println("done")

    // counting
    // done
    // 9876543210
    ```
***

# 数据结构
## 指针
  - Go 具有指针，保存了变量的内存地址，与 C 不同的是，Go **没有指针运算**
    - **\*T** 表示指向类型 T 的值的指针，零值是 `nil`
    - **&** 生成一个指向其作用对象的指针
    - **\*** 表示指针指向的值
    ```go
    var p *int
    i, j := 42, 2701
    p = &i
    fmt.Println(*p, *p+1)
    // 42, 43

    q := &j
    fmt.Println(*q / 37)
    // 73, 3
    ```
## struct 结构体
  - **struct 结构体** 表示一个字段的集合，结构体字段使用 **点号** 来访问，也可以将结构体赋值给指针
    ```go
    type Vertex struct {
        X int
        Y int
    }

    v := Vertex{1, 2}
    v.X = 4
    fmt.Println(v)
    // {4 2}

    p := &v
    p.X = 1e9
    fmt.Println(v)
    // {1000000000 2}
    ```
  - **`name`: `value` 结构体初始化文法** 初始化一个结构体的部分字段
    ```go
    type Vertex struct {
        X, Y int
    }

    var (
        v1 = Vertex{1, 2}
        v2 = Vertex{X: 1} // Y: 0
        v3 = Vertex{} // X: 0, Y: 0
        p  = &Vertex{1, 2}
    )
    fmt.Println(v1, p, v2, v3)
    // {1 2} &{1 2} {1 0} {0 0}
    ```
## 数组
  - **`var / const` 变量名 [长度] 类型** 定义一个数组，数组的长度是其类型的一部分，因此不能改变大小
    ```go
    var a [2]string
    a[0] = "Hello"
    a[1] = "World"
    fmt.Println(a[0], a[1])
    // Hello World
    fmt.Println(a)
    // [Hello World]
    ```
## slice 切片
  - **slice** 包含长度信息，指向一个序列的值
    ```go
    p := []int{2, 3, 5, 7, 11, 13}
    fmt.Println(p, p[0])
    // [2 3 5 7 11 13] 2

    for i := 0; i < len(p); i++ {
        fmt.Printf("p[%d] == %d\n", i, p[i])
    }
    ```
  - **[low:high] 切片** 选取一个序列中的子序列，包含 `low`，不包含 `high`
    ```go
    fmt.Println(p[1:len(p)-1])
    // [3 5 7 11]
    ```
  - **默认值** slice 的默认值是 `nil`
    ```go
    var z []int
    fmt.Println(z == nil, len(z), cap(z))
    // true 0 0
    ```
  - **make 构造 slice**，可以指定 `长度 len` 与 `容量 cap`，`make(type, len, cap)`
    - **len** 是序列中可用元素的数量，在 **索引** 时会判断长度
    - **cap** 是序列最大可以包含的元素数量，为了让 slice 提供可变长度，方便扩容，在 **[low:high] 切片** 时会判断容量
    - 如果不指定 `cap`，则 `cap == len`
    ```go
    a := make([]int, 5)
    fmt.Printf("%s len=%d cap=%d %v\n", "a", len(a), cap(a), a)
    // a len=5 cap=5 [0 0 0 0 0]

    b := make([]int, 0, 5)
    b[3]  // panic: runtime error: index out of range [3] with length 0
    b[:6] // panic: runtime error: slice bounds out of range [:6] with capacity 5

    c = b[:3]
    fmt.Printf("%s len=%d cap=%d %v\n", "c", len(c), cap(c), c)
    // c len=3 cap=5 [0 0 0]
    d = c[2:]
    fmt.Printf("%s len=%d cap=%d %v\n", "d", len(d), cap(d), d)
    // d len=1 cap=3 [0]
    ```
  - **append** 向 slice 添加元素，`func append(s []T, vs ...T) []T`
    - 如果原数组 s 的 **容量 cap** 足够，则在 s 上添加元素，并返回 s
    - 如果原数组 s 的 **容量 cap** 不够，则创建一个更大的数组，并返回新的数组
    - `append(a, b...)` 表示向 a 中添加 b 的所有元素
    ```go
    var z []int
    zz := append(z, 1)
    fmt.Printf("%d, %d, %v", len(zz), cap(zz), zz)
    // 1, 1, [1]
    zz = append(z, 1, 2)
    fmt.Printf("%d, %d, %v", len(zz), cap(zz), zz)
    // 2, 2, [1 2]
    zz := append(z, 1, 2, 3)
    fmt.Printf("%d, %d, %v", len(zz), cap(zz), zz)
    // 3, 4, [1 2 3]
    zz := append(z, 1, 2, 3, 4)
    fmt.Printf("%d, %d, %v", len(zz), cap(zz), zz)
    // 4, 4, [1 2 3 4]
    ```
    ```go
    a := []string{"John", "Paul"}
    b := []string{"George", "Ringo", "Pete"}
    c := append(a, b...)
    fmt.Println(len(c), cap(c), c)
    // 5 5 [John Paul George Ringo Pete]
    ```
  - **copy** 复制序列，`func copy(dst, src []T) int`，返回复制的元素数量
    ```go
    s := [3]string{"a", "b", "c"}
    t := make([]string, len(s), (cap(s)+1)*2) // +1 in case cap(s) == 0
    copy(t, s)  // 1:6: invalid argument: copy expects slice arguments; found t (variable of type []string) and s (variable of type [3]string)
    copy(t, s[:]) // 3
    fmt.Println(len(s), cap(s), s)
    // 3 3 [a b c]
    fmt.Println(len(t), cap(t), t)
    // 3 8 [a b c]
    ```
## range 迭代遍历
  - **range** 在 for 循环中对 slice 或者 map 进行迭代遍历，`range` 给出的是 `元素序号, 元素值`，可以通过 **_** 忽略不用的部分
    ```go
    var pow = []int{1, 2, 4}

    for i, v := range pow {
        fmt.Printf("2**%d = %d\n", i, v)
    }
    // 2**0 = 1
    // 2**1 = 2
    // 2**2 = 4

    for i := range pow {  // for i, _ := range pow {
        fmt.Println(i, pow[i])
    }
    // 0 1
    // 1 2
    // 2 4

    for _, v := range pow {
        fmt.Println(v)
    }
    // 1
    // 2
    // 4
    ```
## map 字典
  - **map** 键值对映射，map 必须用 **make** 来创建，使用 `new` 创建的 map 值为 **nil**，不能赋值
    ```go
    type Vertex struct {
        Lat, Long float64
    }

    var m map[string]Vertex
    fmt.Println(m == nil)
    // true
    m["aa"] = Vertex{1.0, 2.0}
    // panic: assignment to entry in nil map

    m := make(map[string] Vertex)
    fmt.Println(m == nil)
    // false
    m["aa"] = Vertex{1.0, 2.0}
    fmt.Println(m, m["aa"], m["aa"].Lat)
    // map[aa:{1 2}] {1 2} 1
    ```
    ```go
    var mm = map[string] Vertex {
        "aa": Vertex {1.0, 2.0},
        "bb": Vertex {2.0, 3.0},
    }

    var nn = map[string] Vertex {
        "aa": {1.0, 2.0},
        "bb": {2.0, 3.0},
    }
    ```
  - **map 元素操作**
    - **增加 / 修改元素** `m[key] = elem`
    - **删除元素** `delete(m, key)`
    - **双赋值检测某个键是否存在** `elem, ok = m[key]`，如果 `key` 在 `m` 中，`ok` 为 true，否则为 `false`，且 `elem` 为 **map 元素类型的零值**
    - 取不存在的键值时为零值，`m[not_a_key] == 0`
    ```go
    m := make(map[string]int)
    m["Answer"] = 42
    fmt.Println("The value:", m["Answer"])
    // The value: 42
    fmt.Println("The value:", m["Question"])
    // The value: 0

    delete(m, "Answer")
    fmt.Println("The value:", m["Answer"])
    // The value: 0

    v, ok := m["Answer"]
    fmt.Println("The value:", v, "Present?", ok)
    // The value: 0 Present? false
    ```
***

# 方法和接口
## 方法
  - Go 没有类，可以通过在结构体类型上定义方法，实现类似 **类** 的调用，`func (方法接收者) 方法名(参数列表) 返回值 { ... }`
    - 方法接收者不要求是结构体，可以针对 **任意类型** 定义方法，但不能对来自其他包的类型或基础类型定义方法
    - 方法接收者定义为指针，可以避免在每个方法调用中拷贝值，可以在方法中修改接收者的值
    ```go
    import "math"
    type Vertex struct {
        X, Y float64
    }
    func (v * Vertex) Abs() float64 {
        return math.Sqrt(v.X * v.X + v.Y * v.Y)
    }

    vv := &Vertex{3, 4}
    fmt.Println(vv.Abs)
    // 0x7f3520669e80
    ```
    ```go
    type MyFloat float64

    func (f MyFloat) Abs() float64 {
        if f < 0 {
            return float64(-f)
        }
        return float64(f)
    }

    ff := MyFloat(-math.Sqrt2)
    fmt.Println(ff.Abs())
    // 1.4142135623730951
    ff.Scalev()
    fmt.Println(ff)
    ```
  - **方法接收者定义为指针** 此时方法接收到的是 **接收者的原值**，否则接收到的是 **接收者的副本**
    ```go
    type Vertex struct {
        X, Y float64
    }

    func (v Vertex) Scale_v(f float64) {
        v.X = v.X * f
        v.Y = v.Y * f
    }
    func (v * Vertex) Scale_p(f float64) {
        v.X = v.X * f
        v.Y = v.Y * f
    }

    vv := Vertex{3, 4}
    vv.Scale_v(5.)
    fmt.Println(vv)
    // {3 4}
    vv.Scale_p(5.)
    fmt.Println(vv)
    // {15 20}
    (&vv).Scale_p(5.)
    fmt.Println(vv)
    // {75 100}
    pp := &vv
    pp.Scale_p(5.)
    fmt.Println(vv)
    // {375 500}

    vv := &Vertex{3, 4}
    vv.Scale_p(5.)
    fmt.Println(vv)
    // &{15 20}
    vv.Scale_v(5.)  // This could result in problem [ ??? ]
    fmt.Println(vv)
    // &{15 20}

    vv.Scale_v(5.)
    fmt.Println(vv)
    // &{15 20}

    vv.Scale_v(5.)
    fmt.Println(vv)
    // &{5e-324 5e-324} // [ ??? ]
    ```
## 接口
  - **接口** 是由一组方法定义的集合，`接口类型的值` 可以存放实现这些方法的任何值
    - Go 的接口为 **隐式接口**，类型通过实现指定的方法来实现接口，没有显式声明的必要
    - 隐式接口解藕了实现接口的包和定义接口的包，两者互不依赖
    ```go
    // 定义接口类型 Abser
    type Abser interface {
        Abs() float64
    }

    // 实现接口的方法 MyFloat.Abs
    type MyFloat float64

    func (f MyFloat) Abs() float64 {
        if f < 0 {
            return float64(-f)
        }
        return float64(f)
    }

    // 实现接口的方法 *Vertex.Abs
    type Vertex struct {
        X, Y float64
    }

    func (v *Vertex) Abs() float64 {
        return math.Sqrt(v.X*v.X + v.Y*v.Y)
    }

    // 接口类型
    var a Abser
    f := MyFloat(-math.Sqrt2)
    v := Vertex{3, 4}
    a = f
    a = &v
    a = v // Vertex 没有实现 Abser，cannot use v (variable of type exec281.Vertex) as exec279.Abser value in assignment: missing method Abs

    fmt.Println(a.Abs())
    // 5
    ```
    ```go
    type Reader interface {
        Read(b []byte) (n int, err error)
    }

    type Writer interface {
        Write(b []byte) (n int, err error)
    }

    type ReadWriter interface {
        Reader
        Writer
    }
    ```
## Stringers 接口
  - **Stringers 接口** 一个普遍存在的接口是 `fmt` 包中定义的 `Stringer`，可以定义一个 **描述字符串**，`fmt` 等包用来进行输出
    ```go
    type Stringer interface {
        String() string
    }
    ```
    ```go
    type Person struct {
        Name string
        Age  int
    }

    func (p Person) String() string {
        return fmt.Sprintf("%v (%v years)", p.Name, p.Age)
    }

    a := Person{"Arthur Dent", 42}
    z := Person{"Zaphod Beeblebrox", 9001}
    fmt.Println(a, z)
    // Arthur Dent (42 years) Zaphod Beeblebrox (9001 years)
    ```
## Error 错误接口
  - **error 类型** 是一个内建接口，Go 使用 `error` 值来表示错误状态，通常函数会返回一个 error 值，一般 error 为 `nil` 时表示成功，否则表示出错
    ```go
    type error interface {
        Error() string
    }
    ```
    ```go
    import "strconv"
    i, err := strconv.Atoi("42")
    if err != nil {
        fmt.Printf("couldn't convert number: %v\n", err)
    }
    fmt.Println("Converted integer:", i)
    ```
  - **自定义 error**
    ```go
    import "time"

    type MyError struct {
        When time.Time
        What string
    }

    func (e *MyError) Error() string {
        return fmt.Sprintf("at %v, %s", e.When, e.What)
    }

    func run() error {
        return &MyError{time.Now(), "it didn't work"}
    }

    if err := run(); err != nil {
        fmt.Println(err)
    }
    // at 2020-01-03 15:38:38.157514149 +0800 CST m=+105783.750721763, it didn't work
    ```
## Readers 接口
  - [io 包](https://golang.org/pkg/io/)中定义了 **io.Reader 接口**，表示 **从数据流结尾读取数据**
    ```go
    func (T) Read(b []byte) (n int, err error)
    ```
    Read 用数据填充指定的序列 slice，并且返回 **填充的字节数** 和 **错误信息**，在遇到数据流结尾时，返回 **io.EOF 错误**
    ```go
    import (
        "fmt"
        "io"
        "strings"
    )

    r := strings.NewReader("Hello, Reader!")
    b := make([]byte, 8)
    for {
        n, err := r.Read(b)
        fmt.Printf("n = %v err = %v b = %v\n", n, err, b)
        fmt.Printf("b[:n] = %q\n", b[:n])
        if err == io.EOF {
            break
        }
    }
    // n = 8 err = <nil> b = [72 101 108 108 111 44 32 82]
    // b[:n] = "Hello, R"
    // n = 6 err = <nil> b = [101 97 100 101 114 33 32 82]
    // b[:n] = "eader!"
    // n = 0 err = EOF b = [101 97 100 101 114 33 32 82]
    // b[:n] = ""
    ```
  - Go 标准库包含了这个接口的许多 [实现](https://golang.org/search?q=Read#Global)，包括文件 / 网络连接 / 压缩 / 加密等
  - 一个常见模式是 `io.Reader` 包裹另一个 `io.Reader`，然后通过某种形式修改数据流，如 `gzip.NewReader` 函数接受压缩的数据流 `io.Reader`，并且返回同样实现了 `io.Reader` 的解压缩后的数据流 `*gzip.Reader`
## Web 服务器
  - [包 http](https://golang.org/pkg/net/http/) 通过任何实现了 **`http.Handler`** 的值来响应 HTTP 请求
    ```go
    package http

    type Handler interface {
        ServeHTTP(w ResponseWriter, r *Request)
    }
    ```
    类型 `Hello` 实现了 `http.Handler`，访问 [http://localhost:4000/](http://localhost:4000/) 打开页面
    ```go
    import (
        "fmt"
        "log"
        "net/http"
    )

    type Hello struct{}

    func (h Hello) ServeHTTP(w http.ResponseWriter, r *http.Request) {
        fmt.Fprint(w, "Hello!")
    }

    var h Hello
    err := http.ListenAndServe("localhost:4000", h)
    if err != nil {
        log.Fatal(err)
    }
    ```
  - **http.Handle** 在 web 服务器中注册指定路径的处理方法，注册两个路径
    - [http://localhost:4000/string](http://localhost:4000/string)
    - [http://localhost:4000/struct](http://localhost:4000/struct)
    ```go
    import (
        "log"
        "net/http"
        "fmt"
    )
    type String string

    func (ss String) ServeHTTP(w http.ResponseWriter, r *http.Request) {
        fmt.Fprint(w, ss)
    }

    type Struct struct {
        Greeting string
        Punct    string
        Who      string
    }

    func (ss * Struct) String() string {
        return fmt.Sprintf("%v %v %v", ss.Greeting, ss.Punct, ss.Who)
    }

    func (ss * Struct) ServeHTTP(w http.ResponseWriter, r *http.Request) {
        fmt.Fprint(w, ss)
    }

    http.Handle("/string", String("I'm a frayed knot."))
    http.Handle("/struct", &Struct{"Hello", ":", "Gophers!"})
    log.Fatal(http.ListenAndServe("localhost:4000", nil))
    ```
## Image 图片接口
  - [包 image](https://golang.org/pkg/image/#Image) 定义了 **Image 接口**
    - `Bounds` 方法的 `Rectangle` 返回值实际上是一个 `image.Rectangle`， 其定义在 `image` 包中
    - `color.Color` 和 `color.Model` 也是接口，但是通常因为直接使用预定义的实现 `image.RGBA` 和 `image.RGBAModel` 而被忽视了
    - [image/color 包](https://golang.org/pkg/image/color/)
    ```go
    package image

    type Image interface {
        ColorModel() color.Model
        Bounds() Rectangle
        At(x, y int) color.Color
    }
    ```
    ```go
    import (
        "fmt"
        "image"
    )

    m := image.NewRGBA(image.Rect(0, 0, 100, 100))
    fmt.Println(m.Bounds())
    // (0,0)-(100,100)
    fmt.Println(m.At(0, 0).RGBA())
    // 0 0 0 0
    ```
  - **自定义的 Image 类型** 需要实现必要的方法，并且调用 `pic.ShowImage`
    - **Bounds** 应当返回一个 `image.Rectangle`，如 `image.Rect(0, 0, w, h)`
    - **ColorModel** 应当返回 `color.RGBAModel`
    - **At** 应当返回一个颜色，如 `color.RGBA{v, v, 255, 255}`
***

# 并发
## goroutine
  - **goroutine** 是由 Go 运行时环境管理的轻量级线程，goroutine 在相同的地址空间中运行，因此访问共享内存必须进行同步，`sync` 提供了这种可能，不过在 Go 中并不经常用到，因为有其他的办法
    ```go
    // 开启一个新的 goroutine 执行，f / x / y / z 是当前 goroutine 中定义的，但是在新的 goroutine 中运行 f
    go f(x, y, z)
    ```
    ```go
    import "time"

    func say(s string) {
        for i := 0; i < 5; i++ {
            time.Sleep(100 * time.Millisecond)
            fmt.Println(s)
        }
    }

    go say("world")
    say("hello")
    // world
    // hello
    // world
    // hello
    // hello
    // world
    // world
    // hello
    // world
    // hello
    ```
## channel
  - **channel** 是有类型的管道
    - 可以用 channel 操作符 `<-` 对其发送或者接收值
    - channel 使用前 **必须创建**
    - 默认情况下，在另一端准备好之前，发送和接收都会阻塞，这使得 goroutine 可以在没有明确的锁或竞态变量的情况下进行同步
    ```go
    ch := make(chan int)
    ch <- v    // 将 v 送入 channel ch
    v := <-ch  // 从 ch 接收，并且赋值给 v
    ```
    ```go
    func sum(a []int, c chan int) {
        sum := 0
        for _, v := range a {
            sum += v
        }
        c <- sum // 将和送入 c
    }

    a := []int{7, 2, 8, -9, 4, 0}
    c := make(chan int)
    go sum(a[:len(a)/2], c)
    go sum(a[len(a)/2:], c)
    x, y := <-c, <-c // 从 c 中获取
    fmt.Println(x, y, x+y)
    // -5 17 12
    ```
  - **缓冲 channel** `ch := make(chan int, 100)`，指定 **缓冲长度** 来初始化一个缓冲 channel
    - 发送数据时，在缓冲区满的时候会阻塞
    - 接受数据时，在缓冲区清空的时候阻塞
    ```go
    c := make(chan int, 2)
    c <- 1
    c <- 2
    fmt.Println(<-c)
    fmt.Println(<-c)
    // 1
    // 2
    // 2
    ```
  - **close 关闭 channel**
    - **发送者** 可以调用 close 关闭一个 channel，表示没有值会再被发送
    - **接收者** 可以通过赋值语句的第二个参数来测试 channel 是否被关闭，`v, ok := <-ch`，channel 关闭时 `ok == false`
    - 向一个已经关闭的 channel 发送数据会引起 panic
    - 通常情况下无需关闭 channel，只有在需要告诉接收者没有更多的数据的时候才有必要进行关闭，如中断一个 `range`
  - **range 循环接收** `for i := range c` 会不断从 channel 接收值，直到被 `close` 关闭
    ```go
    func fibonacci(n int, c chan int) {
        x, y := 0, 1
        for i := 0; i < n; i++ {
            c <- x
            x, y = y, x+y
        }
        close(c)
    }

    c := make(chan int, 10)
    go fibonacci(cap(c), c)
    for i := range c {
        fmt.Printf("%v ", i)
    }
    // 0 1 1 2 3 5 8 13 21 34
    ```
## select
  - **select** 使一个 `goroutine` 在多个通讯操作上等待
    - `select` 会阻塞，直到条件分支中的某个可以继续执行，这时就会执行那个条件分支
    - 当多个都准备好时，会 **随机选择** 一个
    ```go
    func fibonacci(c, quit chan int) {
        x, y := 0, 1
        for {
            select {
            case c <- x:
                x, y = y, x+y
            case <-quit:
                fmt.Println("quit")
                return
            }
        }
    }

    c := make(chan int)
    quit := make(chan int)
    go func() {
        for i := 0; i < 10; i++ {
            fmt.Printf("%v ", <-c)
        }
        quit <- 0
    }()
    fibonacci(c, quit)
    // 0 1 1 2 3 5 8 13 21 34 quit
    ```
  - **默认选择** 为了 **非阻塞** 的发送或者接收，可使用 `default` 分支，当 select 中的其他条件分支都没有准备好的时候，`default` 分支会被执行
    ```go
    select {
    case i := <-c:
        // 使用 i
    default:
        // 从 c 读取会阻塞
    }
    ```
    ```go
    import "time"

    tick := time.Tick(100 * time.Millisecond)
    boom := time.After(500 * time.Millisecond)
    for {
        select {
        case <-tick:
            fmt.Printf("tick")
        case <-boom:
            fmt.Println(" BOOM!")
            return
        default:
            fmt.Printf(".")
            time.Sleep(50 * time.Millisecond)
        }
    }
    // ..tick..tick..tick..tick..tick BOOM!
    ```
***
