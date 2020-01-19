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
  	- [约定规则](#约定规则)
  	- [Go 程序的一般结构](#go-程序的一般结构)
  	- [import](#import)
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
  - [The Go Playground](https://play.golang.org/)
  - [Effective Go](https://golang.org/doc/effective_go.html)
  - [Effective Go中文版](https://go-zh.org/doc/effective_go.html)
  - [Effective Go中文版](https://www.kancloud.cn/kancloud/effective/72199)
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
    ```sh
    ''' Q
    go Get https://proxy.golang.org/xxx proxyconnect tcp: net/http: TLS handshake timeout
    '''
    ''' A
    配置 https 代理为 http://xxx

    http_proxy='http://127.0.0.1:8118' https_proxy='http://127.0.0.1:8118' jupyter-notebook --notebook-dir=$HOME/practice_code/jupyter_notebook/
    '''
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
## gofmt 格式化代码
  - **参数**
    - **-s** 开启简化代码，如去除不必要的类型声明，去除迭代时非必要的变量赋值等
    - **-d** 显示 diff，而不是转化后的结果
    - **-w** 覆盖源文件
    - **-l ./*.go** 列出需要格式化的文件
    - **-r 'a[b:len(a)] -> a[b:]'** 指定替换规则
  - **示例**
    ```sh
    gofmt -s -d test.go
    gofmt -s -w test.go

    gofmt -s -l ./*.go
    gofmt -s -d ./*.go
    gofmt -s -w ./*.go
    ```
## go doc 帮助文档
  ```sh
  go doc -all fmt | grep -i printf

  go doc -all fmt.Println
  go doc -src fmt.Println
  ```
***

# 基础语法
## 约定规则
  - **命名规则**
    - 一个名字在程序包之外的可见性是由它的首字符 **是否为大写** 来确定的
    - 按照约定，程序包使用小写，尽量是一个单词的名字，不需要使用下划线或者混合大小写，不用担心会与先前的有冲突，程序包名只是导入的缺省名字
    - 按照约定，程序包名为其源目录的基础名
    - 按照约定，单个方法的接口使用方法名加上“er”后缀来命名，如果类型实现的是一个和众所周知的类型具有相同含义的方法，那么就使用相同的名字和签名
    - Go 约定使用 `MixedCaps` 或者 `mixedCaps` 的形式，而不是下划线来书写多个单词的名字
  - **注释规则**
    - 每个程序包都应该有一个 **包注释**，位于 package 子句之前的块注释，对于有多个文件的程序包，包注释只需要出现在一个文件中，任何一个文件都可以
    - **包注释** 应该用来介绍该程序包，并且提供与整个程序包相关的信息，它将会首先出现在 `godoc` 页面上，并会建立后续的详细文档
    - 在程序包里面，任何直接位于顶层声明之前的注释，都会作为该 **声明的文档注释**，程序中每一个被导出的名字，都应该有一个文档注释
    - **文档注释** 作为完整的语句可以工作的最好，可以允许各种自动化的展现，第一条语句应该为一条概括语句，并且 **使用被声明的名字作为开头**
  - **分号**
    - Go 的规范语法是使用 **分号** 来终结语句的，但这些分号并不在源码中出现，词法分析器会在扫描时，使用简单的规则自动插入分号
    - 分号插入规则所导致的一个结果是，不能将控制结构 `if` / `for` / `switch` / `select` 的左大括号放在下一行，如果这样做，则会在大括号之前插入一个分号
  - **空白标识符 _**
    - **多赋值语句** 需要多个左值，但如果其中某个左值在程序中并没有被使用到，那么就需要用 **空白标识符** 来占位，以避免引入一个新的无用变量
    - **未使用的导入和变量** 如果在程序中导入了一个包或声明了一个变量，却没有使用的话，会引起编译错误，可以将包 / 变量赋值给 `_` 禁止编译错误
    - **只调用导入包的 init 函数** 有时需要导入一些包，但不实际使用，只是调用其中的 `init` 函数
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
  - **项目目录** 一般包含三个文件夹，分别为 `src` / `pkg` / `bin`
    - **src** 存放 golang 源码
    - **pkg** 存放编译后的文件
    - **bin** 存放编译后可执行的文件
## import
  - **包 package / import** Go 程序是通过 package 来组织的
    - 只有 **package** 名称为 **main** 的包可以包含 main 函数，一个可执行程序有且仅有一个 main 包
    - 通过 **import** 关键字来导入其他非 main 包，使用 `<PackageName>.<FunctionName>` 调用
    - 文件名 / 文件夹名与包名没有直接关系，不需要一致，但按照惯例，最好一致，同一个文件夹下的文件只能有一个包名，否则编译报错
    - 可以使用 **()** 打包导入多个
    ```go
    package main  // 当前程序的包名
    import "fmt"  // 导入其他包
    import (  // 同时导入多个
        "fmt"
        "math/rand"
    )
    import fmt2 "fmt" // package 别名
    import . "fmt"  // 表示省略调用，调用该模块里面的函数可以不写模块名
    ```
  - **相对导入** 导入当前文件夹下的某个 module 的文件夹
    ```go
    import (
        "./test1"
        "../test2"
    )
    ```
  - **绝对导入** 将目标项目添加到 `$GOPATH` 环境变量中
    ```sh
    export GOPATH=$GOPATH:$HOME/practice_code/go
    ```
    ```go
    import (
        "project/module1"
        "project/module2/t"
    )
    ```
  - **别名** 重命名导入的包
    ```go
    import f "fmt"
    f.Println("test")
    ```
  - **.** 导入包后，调用时省略包名
    ```go
    import . "fmt"
    Println("test")
    ```
  - **_** 导入包，但不直接使用包中的函数，而是调用包中的 `init` 函数
    ```go
    // module/module1.go
    package module1
    import "fmt"

    func init() {
       fmt.Println("this is module1")
    }
    ```
    ```go
    // main.go
    package main
    import (
        "fmt"
        _ "module"
    )

    func main() {
        fmt.Println("this is a test")
    }
    ```
    **Run**
    ```sh
    $ go run main.go
    this is module1
    this is a test
    ```
## Go 数据类型
  - Go语言中，使用 **大小写** 来决定该常量、变量、类型、接口、结构或函数是否可以被外部包所调用，即 private / public
  - Go 中的字符串只能使用 **双引号**
    ```go
    aa := "aa"
    fmt.Printf("%T", aa)
    // string6
    ```
  - **数据定义 const / var / type**
    - Go 语言的 **类型** 在 **变量名之后**
    - `var` 语句可以定义在包或函数级别，即在函数外或函数内
    - 变量在没有初始化时默认为 **零值**，数值类型为 `0`，布尔类型为 `false`，字符串为 `""` 空字符串
    - **const** 关键字定义常量，常量是在 **编译时** 被创建，即使是函数内部的局部常量，常量的表达式必须为能被编译器求值的常量表达式
    - **var** 关键字定义变量，在函数体外部使用则定义的是 **全局变量**，初始值可以为 **运行时** 计算的通用表达式
    - **type** 关键字定义一般类型，如结构 struct / 接口 interface
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
  - 在 `:=` 声明中，变量 `v` 即使已经被声明过，也可以出现，前提是
    - 该声明和 v 已有的声明在相同的作用域中，如果 v 已经在外面的作用域里被声明了，则该声明将会创建一个新的变量
    - 初始化中相应的值是可以被赋给 v 的
    - 并且，声明中 **至少有其它一个变量** 将被声明为一个新的变量
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
    其中 `rune` 是 Go 的术语，用于指定一个 **单独的 Unicode 编码点**
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
  - **type 定义的类型转化** 对于 type 定义的类型，如果忽略类型名，两个类型是相同的，则类型转换是合法的，该转换并不创建新值，只是暂时使现有的值具有一个新的类型
    ```go
    type Sequence []int
    aa := Sequence{1, 3, 5, 2, 4}
    bb := []int(aa)
    fmt.Printf("%T, %T", aa, bb)
    // []int, []int
    ```
  - **interface {} 类型** 在函数定义与签名中，`interface {}` 类型可以用于接受 / 返回任意类型
    ```go
    func interfier(ii interface{}) interface{} {
        return ii
    }

    bb := interfier([]int{1, 2, 3})
    fmt.Printf("%T", bb) // []int
    ```
    类型转化使用 `(interface {}).(type)`，如果实际类型与 `type` 相同，则返回 `value, true`，否则返回 `zero_velue, false`
    ```go
    []int(bb) // cannot convert interface{} to []int: bb

    bb.([]int)  // [1 2 3] true
    bb.([4]int) // [0 0 0 0] false

    str, ok := bb.(string)
    if ok {
        fmt.Printf("string value is: %q\n", str)
    } else {
        fmt.Printf("value is not a string\n")
    }
    // value is not a string
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
  - Go 没有逗号操作符，因此如果需要在 for 中运行多个变量，需要使用并行赋值，并且不能使用 `++` / `--` 操作
    ```go
    // Reverse a
    a := []int{1, 2, 3, 4, 5}

    // NOT: for i, j := 0, len(a)-1; i < j; j++, i-- {
    for i, j := 0, len(a)-1; i < j; i, j = i+1, j-1 {
        a[i], a[j] = a[j], a[i]
    }
    fmt.Println(a)
    // [5 4 3 2 1]
    ```
  - 在嵌套的 `for` 循环中，可以使用 `break` / `continue` 加一个 **标号** 来指定跳出的是 **哪一个循环**
    ```go
    sum := 0
    Loop1:
    for {
        for i := 0; i < 5; i++ {
            fmt.Printf("%d ", i)
            sum += i
            if sum > 5 {
                break Loop1
            }   
        }   
    }   
    fmt.Println(sum)
    // 0 1 2 3 6
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
  - **逻辑表达式** 使用 `||` / `&&` / `!`
  - 当 `if` 语句不会流向下一条语句时，如控制结构体结束于 `break` / `continue` / `goto` / `return`，则通常是省略掉 `else`
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
  - **没有条件的 switch** 则对为 `true` 的条件进行匹配，可以用更清晰的形式编写长的 `if-else` 链
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
  - **case** 可以使用 **逗号分隔的列表**
    ```go
    func shouldEscape(c byte) bool {
        switch c {
        case ' ', '?', '&', '=', '#', '+', '%':
            return true
        }
        return false
    }
    ```
  - **switch 动态类型判断** 可以用于获得一个 **接口变量** 的动态类型，在括号中使用关键字 **type**
    ```go
    func interfier(ii interface{}) interface{} {
        return ii
    }

    var t interface{}
    t = interfier(true)

    switch t := t.(type) {
    default:
        fmt.Printf("unexpected type %T", t)       // %T prints whatever type t has
    case bool:
        fmt.Printf("boolean %t\n", t)             // t has type bool
    case int:
        fmt.Printf("integer %d\n", t)             // t has type int
    case *bool:
        fmt.Printf("pointer to boolean %t\n", *t) // t has type *bool
    case *int:
        fmt.Printf("pointer to integer %d\n", *t) // t has type *int
    }
    // boolean true
    ```
## func 函数
  - **函数声明 func**，函数可以没有参数或接受多个参数，类似于变量定义，返回值类型在函数名之后
    ```go
    func main(argc int, argv []string) int { ... }

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
  - **命名的结果参数** 函数的返回参数可以给定一个名字，并在函数内作为一个普通变量来使用
    - 当被命名时，它们在函数起始处被初始化为对应类型的零值
    - 如果函数执行了没有参数的 `return` 语句，则返回 **函数内对应参数的当前值**
    ```go
    func nextInt(b []byte, pos int) (value, nextPos int) { ... }
    ```
    ```go
    // io.ReadFull
    func ReadFull(r Reader, buf []byte) (n int, err error) {
        for len(buf) > 0 && err == nil {
            var nr int
            nr, err = r.Read(buf)
            n += nr
            buf = buf[nr:]
        }
        return
    }
    ```
## defer 延迟调用
  - **defer 语句** 延期执行，使其在执行 defer 的函数即将返回之前才被运行，延迟调用的 **参数会立刻生成**，可以用于释放资源等，如释放互斥锁或关闭文件
    ```go
    i := 0
    defer fmt.Println(i)
    i++
    // 0
    ```
  - **defer 栈 LIFO** 延迟的函数调用被压入一个栈中，当函数返回时，会按照后进先出的顺序调用被延迟的函数调用
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
## init 函数
  - **init 初始化函数** 每个源文件可以定义一个或多个不带参数的 `(niladic)init` 函数
    - **init** 是在程序包以及导入的程序包中，所有变量声明都被初始化后才被调用
    - 除了用于无法通过声明来表示的初始化以外，init 函数的一个常用法是在真正执行之前进行验证或者修复程序状态的正确性
    ```go
    func init() {
        if user == "" {
            log.Fatal("$USER not set")
        }
        if home == "" {
            home = "/home/" + user
        }
        if gopath == "" {
            gopath = home + "/go"
        }
        // gopath may be overridden by --gopath flag on command line.
        flag.StringVar(&gopath, "gopath", gopath, "override default GOPATH")
    }
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
  - Go 语言中返回一个 **局部变量的地址** 是绝对没有问题的，变量关联的存储在函数返回之后依然存在
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
  - Go 语言中的数组是 **值**，而不是指针
  - **`var / const` 变量名 [长度] 类型** 定义一个数组
    - 数组的长度是其类型的一部分，因此不能改变大小，`[10]int` 和 `[20]int` 是不同的
    - 同类型数组间的 `=` 操作会拷贝所有的元素，在函数中作为参数传递时，也是 **值传递**
    - 对于数组的操作，通常使用 **切片 slice**
    ```go
    var a [2]string
    a[0] = "Hello"
    a[1] = "World"
    fmt.Println(a[0], a[1])
    // Hello World
    fmt.Println(a)
    // [Hello World]

    b := []string {"a", "b", "c"}
    fmt.Printf("%T, %T", a, b)
    // [2]string, []string
    ```
  - **`index`: `value` 初始化文法** 初始化指定位置的值
    ```go
    a := [3]int {1:3}
    b := []int {1:3}
    fmt.Printf("%T, %v, %T, %v", a, a, b, b)
    // [3]int, [0 3 0], []int, [0 3]
    ```
## slice 切片
  - **slice 切片**
    - **切片** 对数组进行封装，提供了一个针对串行数据，更加通用，强大和方便的接口，除了像转换矩阵这样具有显式维度的项，Go 中大多数的数组编程都是通过切片完成
    - **切片** 包含 **len 长度** / **cap 容量** 信息，以及一个 **底层数组的引用**
    ```go
    p := []int{2, 3, 5, 7, 11, 13}
    fmt.Println(p, p[0])
    // [2 3 5 7 11 13] 2

    for i := 0; i < len(p); i++ {
        fmt.Printf("p[%d] == %d\n", i, p[i])
    }
    ```
  - **切片间的赋值**
    - 同类型的两个切片间赋值，将指向同一个底层数组，作为函数参数时，对切片元素的修改，对于调用者是可见的，类似于 **指针传递**
    - 切片本身是按照 **值传递** 的，即传递切片的 **底层数组指针** / **长度** / **容量** 运行时数据结构
    ```go
    p := []int{2, 3, 5, 7, 11, 13}
    func foo(tt []int) {
        tt[0], tt[len(tt)-1] = tt[len(tt)-1], tt[0]
    }
    foo(p)
    p  // [13 3 5 7 11 2]
    ```
  - **[low:high] 切片** 选取一个切片中的子切片，包含 `low`，不包含 `high`
    ```go
    fmt.Println(p[1:len(p)-1])
    // [3 5 7 11]
    ```
  - **零值** slice 的零值是 `nil`，但 `len` / `cap` 调用是合法的，并且返回 `0`
    ```go
    var z []int
    fmt.Println(z == nil, len(z), cap(z))
    // true 0 0
    ```
  - **make 构造 slice**，可以指定 `长度 len` 与 `容量 cap`，`make(type, len, cap)`
    - **len** 是切片中可用元素的数量，在 **索引** 时会判断长度
    - **cap** 是切片最大可以包含的元素数量，为了让 slice 提供可变长度，方便扩容，在 **[low:high] 切片** 时会判断容量
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
    - 如果原切片 s 的 **容量 cap** 足够，则在 s 上添加元素，并返回 s
    - 如果超出了原切片 s 的 **容量 cap**，则切片会被重新分配，然后返回新产生的切片
    ```go
    func append(slice []T, elements ...T) []T
    ```
    其中 `T` 为任意给定类型的占位符，在 Go 中是无法写出一个类型 T 由调用者来确定的函数的，因此 append 是内建函数，它需要编译器的支持
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
    `append(a, b...)` 表示向 a 中添加 b 的所有元素
    ```go
    a := []string{"John", "Paul"}
    b := []string{"George", "Ringo", "Pete"}
    c := append(a, b...)
    fmt.Println(len(c), cap(c), c)
    // 5 5 [John Paul George Ringo Pete]
    ```
  - **copy** 复制切片，`func copy(dst, src []T) int`，返回复制的元素数量，可以用于类型转化，将非固定长度的 **slice** 转化为固定长度的 **array**
    ```go
    s := [3]string{"a", "b", "c"}
    t := make([]string, len(s), (cap(s)+1)*2) // +1 in case cap(s) == 0
    fmt.Printf("%T, %T", s, t)
    // [3]string, []string

    // Error: incompatible types in assignment: []string = [3]string
    t = s

    // Error: second argument to copy should be slice or string; have s <[3]string>
    copy(t, s)

    copy(t, s[:]) // 3
    fmt.Println(len(s), cap(s), s)
    // 3 3 [a b c]
    fmt.Println(len(t), cap(t), t)
    // 3 8 [a b c]
    ```
## range 迭代遍历
  - **range** 在 for 循环中对 `slice` 或者 `map` 进行迭代遍历
    - `range` 给出的是 **元素序号, 元素值**，`for key, value := range array`
    - 如果只需要 `range` 中的第一项，则可以丢弃第二个，`for key := range m`
    - 如果只需要 `range` 中的第二项，则可以使用空白标识符 **_** ，来丢弃第一个，`for _, value := range array`
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
  - **字符串的 range 操作**，会通过解析 `UTF-8` 来拆分出 **单个的 Unicode 编码点**，错误的编码会消耗一个字节，产生一个替代的 **符文 rune `U+FFFD`**
    ```go
    import "fmt"
    for pos, char := range "日本\x80語" { // \x80 is an illegal UTF-8 encoding
        fmt.Printf("character %#U starts at byte position %d\n", char, pos)
    }
    // character U+65E5 '日' starts at byte position 0
    // character U+672C '本' starts at byte position 3
    // character U+FFFD '�' starts at byte position 6
    // character U+8A9E '語' starts at byte position 7
    ```
  - **数组类型转化**
    ```go
    a := []int{1, 2, 3, 4, 5}
    int2float32 := func(aa []int) []float32 {
        bb := make([]float32, len(aa))
        for ii, vv := range(aa) {
            bb[ii] = float32(vv)
        }
        return bb
    }
    int2float32(a)
    // [1 2 3 4 5]
    ```
## map 字典
  - **map** 键值对映射
    - map 必须用 **make** 来创建，使用 `new` 创建的 map 值为 **nil**，不能赋值
    - map 的 key 可以为任何定义了 **等于操作符** 的类型，如整数 / 浮点 / 复数 / 字符串 / 指针 / 接口/ 结构体 / 数组，切片不能作为 key，因为没有定义等于操作
    - 和切片类似，map 持有对底层数据结构的引用，传递给函数时，对 map 内容的改变，对调用者是可见的
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
    - **delete 删除元素** `delete(m, key)`
    - 取不存在的键值 key 对应的值时，值为 value 类型对应的零值，`m[not_a_key] == 0`
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
    ```
  - **comma ok 双赋值检测某个键是否存在** `elem, ok = m[key]`
    - 如果 `key` 在 `m` 中，`ok` 为 true
    - 如果 `key` 不在 `m` 中，`ok` 为 `false`，且 `elem` 为 **map 元素类型的零值**
    ```go
    v, ok := m["Answer"]
    fmt.Println("The value:", v, "Present?", ok)
    // The value: 0 Present? false
    ```
    ```go
    func offset(tz string) int {
        if seconds, ok := timeZone[tz]; ok {
            return seconds
        }
        log.Println("unknown time zone:", tz)
        return 0
    }
    ```
## make 与 new
  - Go 有两个分配原语，内建函数 `new` 和 `make`，它们所做的事情有所不同，并且用于不同的类型
  - **new** 不初始化内存，只是将其 **置零**，并返回它的地址，`new(T)` 会为 `T` 类型的新项目分配被置零的存储，并返回 **`*T` 的指针**
    ```go
    aa := new([3]int)
    fmt.Println(aa) // &[0 0 0]

    (*aa)[1] = 2
    fmt.Println(aa) // &[0 2 0]
    ```
    对于 `map` / `slice` 类型，对应的零值为 `nil`
    ```go
    cc := new([]int)
    *cc == nil  // true

    dd := new(map[int]int)
    *dd == nil  // true
    ```
  - **make** 只用来创建 `slice` / `map` / `channel`，并且返回一个 **初始化** 的，类型为 **`T` 的值**
    - 之所以有所不同，是因为这三个类型的背后是象征着，对使用前必须初始化的数据结构的引用
    - 如，`slice` 是一个三项描述符，包含一个指向数据的指针，长度，以及容量，在这些项被初始化之前，`slice` 都是 `nil` 的
    - 对于 `slice` / `map` / `channel`，`make` 初始化内部数据结构，并准备好可用的值
    ```go
    ee := make([]int, 3)
    fmt.Println(ee) // [0 0 0]

    ee[2] = 2
    fmt.Println(ee) // [0 0 2]
    ```
## 枚举常量
  - **枚举常量** 使用 `iota` 枚举器来创建，由于 `iota` 可以为表达式的一部分，并且表达式可以被隐式的重复，所以很容易创建复杂的值集
    ```go
    type ByteSize float64

    const (
        _           = iota // ignore first value by assigning to blank identifier
        KB ByteSize = 1 << (10 * iota)
        MB
        GB
        TB
        PB
        EB
    )
    fmt.Println(KB, MB, GB, TB, PB, EB)
    // 1024 1.048576e+06 1.073741824e+09 1.099511627776e+12 1.125899906842624e+15 1.152921504606847e+18
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
    - 接口为指定对象的行为提供了一种方式，**如果事情可以这样做，那么它就可以在这里使用**
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
    package main
    import "fmt"
    type Person struct {
        Name string
        Age  int
    }

    func (p Person) String() string {
        return fmt.Sprintf("%v (%v years)", p.Name, p.Age)
    }

    func main() {
        a := Person{"Arthur Dent", 42}
        z := Person{"Zaphod Beeblebrox", 9001}
        fmt.Println(a, z)
    }
    // Arthur Dent (42 years) Zaphod Beeblebrox (9001 years)
    ```
  - **String 方法** 通常用在自定义的结构体上，但也可用于标量类型，比如 `ByteSize` 这样的浮点类型
    ```go
    package main
    import "fmt"

    type ByteSize float64

    const (
        _           = iota // ignore first value by assigning to blank identifier
        KB ByteSize = 1 << (10 * iota)
        MB
        GB
        TB
        PB
        EB
        ZB
        YB
    )

    func (b ByteSize) String() string {
        switch {
        case b >= YB:
            return fmt.Sprintf("%.2fYB", b/YB)
        case b >= ZB:
            return fmt.Sprintf("%.2fZB", b/ZB)
        case b >= EB:
            return fmt.Sprintf("%.2fEB", b/EB)
        case b >= PB:
            return fmt.Sprintf("%.2fPB", b/PB)
        case b >= TB:
            return fmt.Sprintf("%.2fTB", b/TB)
        case b >= GB:
            return fmt.Sprintf("%.2fGB", b/GB)
        case b >= MB:
            return fmt.Sprintf("%.2fMB", b/MB)
        case b >= KB:
            return fmt.Sprintf("%.2fKB", b/KB)
        }
        return fmt.Sprintf("%.2fB", b)
    }
    func main() {
        fmt.Println(TB)
        fmt.Println(ByteSize(1e13))
    }
    // 1.00TB
    // 9.09TB
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
## write 接口
  - 一个实现了 `Write` 接口的类型，可以用于 `Fprintf` 的输出
  ```go
  package main

  import "fmt"

  type ByteSlice []byte
  func (p *ByteSlice) Write(data []byte) (n int, err error) {
      slice := *p
      // Again as above.
      slice = append(slice, data...)
      *p = slice
      return len(data), nil
  }

  func (p ByteSlice) String() string {
      return string(p)
  }

  func main() {
      var b ByteSlice
      fmt.Fprintf(&b, "Hello, ")
      fmt.Println(b)
      fmt.Fprintf(&b, "This hour has %d days", 7)
      fmt.Println(b)
  }
  // Hello,
  // Hello, This hour has 7 days
  ```
## sort 接口
  - **sort.Interface** 需要实现 `Len()` / `Less(i, j int) bool` / `Swap(i, j int)` 三个接口
    ```go
    package main
    import (
        "fmt"
        "sort"
    )

    type Sequence []int

    // Methods required by sort.Interface.
    func (s Sequence) Len() int {
        return len(s)
    }
    func (s Sequence) Less(i, j int) bool {
        return s[i] < s[j]
    }
    func (s Sequence) Swap(i, j int) {
        s[i], s[j] = s[j], s[i]
    }

    // Method for printing - sorts the elements before printing.
    func (s Sequence) String() string {
        sort.Sort(s)
        str := "["
        for i, elem := range s {
            if i > 0 {
                str += " "
            }   
            str += fmt.Sprint(elem)
        }   
        return str + "]"
    }

    func main() {
        aa := Sequence{1, 3, 2, 5, 4}
        fmt.Println(aa)
    }
    // [1 2 3 4 5]
    ```
    该示例中，可以直接使用 `sort.IntSlice` 使 `Sequence` 作为 `[]int` 类型用于平自序
    ```go
    aa := Sequence{1, 3, 2, 5, 4}
    sort.IntSlice(aa).Sort()
    fmt.Println(aa)
    ```
## Web 服务器
  - [包 http](https://golang.org/pkg/net/http/) 通过任何实现了 **`http.Handler` 接口** 的值来响应 HTTP 请求
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
## 线程锁
  - 不加锁时 `goroutine` 是线程不安全的
    ```go
    import (
        "fmt"
        "time"
    )

    var count int

    func test1() {
        aa := time.Now()
        for i := 0; i < 1000000; i++ {
            count++
        }
        bb := time.Now()
        fmt.Println("test1: ", float64(bb.Nanosecond() - aa.Nanosecond()) / 1e9)
    }

    func test2() {
        aa := time.Now()
        for i := 0; i < 1000000; i++ {
            count++
        }
        bb := time.Now()
        fmt.Println("test2: ", float64(bb.Nanosecond() - aa.Nanosecond()) / 1e9)
    }

    count = 0
    go test1()
    go test2()

    fmt.Printf("count=%d\n", count)
    // count=1046347
    // test1:  0.037959184
    // test2:  0.038162785
    ```
  - **sync.Mutex 互斥锁** 只能有一个线程获取到资源，其他线程等待，`sync.Mutex` 的零值被定义为一个未加锁的互斥
    ```go
    import "sync"
    var mm sync.Mutex
    func test1() {
        aa := time.Now()
        for i := 0; i < 1000000; i++ {
            mm.Lock()
            count ++
            mm.Unlock()
        }
        bb := time.Now()
        fmt.Println("test1: ", float64(bb.Nanosecond() - aa.Nanosecond()) / 1e9)
    }
    func test2() {
        aa := time.Now()
        for i := 0; i < 1000000; i++ {
            mm.Lock()
            count ++
            mm.Unlock()
        }
        bb := time.Now()
        fmt.Println("test2: ", float64(bb.Nanosecond() - aa.Nanosecond()) / 1e9)
    }

    count := 0
    go test1()
    go test2()

    fmt.Println(count)
    // 2000000
    // test2:  0.298956036
    // test1:  0.322732425
    ```
  - **sync.RWMutex 读写锁**
    - 当一个线程获取到 **写锁 Lock** 后，其他线程读写都会等待
    - 当一个线程获取到 **读锁 RLock** 后，其他线程获取写锁会等待，读锁可以继续获得
    ```go
    var ww sync.RWMutex
    func test1() {
        aa := time.Now()
        for i := 0; i < 1000000; i++ {
            ww.Lock()
            count ++
            ww.Unlock()
        }
        bb := time.Now()
        fmt.Println("test1: ", float64(bb.Nanosecond() - aa.Nanosecond()) / 1e9)
    }
    func test2() {
        aa := time.Now()
        for i := 0; i < 1000000; i++ {
            ww.Lock()
            count ++
            ww.Unlock()
        }
        bb := time.Now()
        fmt.Println("test2: ", float64(bb.Nanosecond() - aa.Nanosecond()) / 1e9)
    }
    count := 0
    go test1()
    go test2()
    fmt.Println(count)
    // 2000000
    // test1:  0.349657722
    // test2:  0.382042907
    ```
***

## 打印输出
  Go中的格式化打印使用了与C中printf家族类似的风格，不过更加丰富和通用。这些函数位于fmt程序包中，并具有大写的名字：fmt.Printf，fmt.Fprintf，fmt.Sprintf等等。字符串函数（Sprintf等）返回一个字符串，而不是填充到提供的缓冲里。

  你不需要提供一个格式串。对每个Printf，Fprintf和Sprintf，都有另外一对相应的函数，例如Print和Println。这些函数不接受格式串，而是为每个参数生成一个缺省的格式。Println版本还会在参数之间插入一个空格，并添加一个换行，而Print版本只有当两边的操作数都不是字符串的时候才增加一个空格。在这个例子中，每一行都会产生相同的输出。
  ```go
  fmt.Printf("Hello %d\n", 23)
  fmt.Fprint(os.Stdout, "Hello ", 23, "\n")
  fmt.Println("Hello", 23)
  fmt.Println(fmt.Sprint("Hello ", 23))
  ```
  格式化打印函数fmt.Fprint等，接受的第一个参数为任何一个实现了io.Writer接口的对象；变量os.Stdout和os.Stderr是常见的实例。

  接下来这些就和C不同了。首先，数字格式，像%d，并不接受正负号和大小的标记；相反的，打印程序使用参数的类型来决定这些属性。
  ```go
  var x uint64 = 1<<64 - 1
  fmt.Printf("%d %x; %d %x\n", x, x, int64(x), int64(x))
  ```
  会打印出
  ```go
  18446744073709551615 ffffffffffffffff; -1 -1
  ```
  如果只是想要缺省的转换，像十进制整数，你可以使用通用格式%v（代表“value”）；这正是Print和Println所产生的结果。而且，这个格式可以打印任意的的值，甚至是数组，切片，结构体和map。这是一个针对前面章节中定义的时区map的打印语句
  ```go
  fmt.Printf("%v\n", timeZone)  // or just fmt.Println(timeZone)
  ```
  其会输出
  ```go
  map[CST:-21600 PST:-28800 EST:-18000 UTC:0 MST:-25200]
  ```
  当然，map的key可能会按照任意顺序被输出。当打印一个结构体时，带修饰的格式%+v会将结构体的域使用它们的名字进行注解，对于任意的值，格式%#v会按照完整的Go语法打印出该值。
  ```go
  type T struct {
      a int
      b float64
      c string
  }
  t := &T{ 7, -2.35, "abc\tdef" }
  fmt.Printf("%v\n", t)
  fmt.Printf("%+v\n", t)
  fmt.Printf("%#v\n", t)
  fmt.Printf("%#v\n", timeZone)
  ```
  会打印出
  ```go
  &{7 -2.35 abc   def}
  &{a:7 b:-2.35 c:abc     def}
  &main.T{a:7, b:-2.35, c:"abc\tdef"}
  map[string] int{"CST":-21600, "PST":-28800, "EST":-18000, "UTC":0, "MST":-25200}
  ```
  （注意符号&）还可以通过%q来实现带引号的字符串格式，用于类型为string或[]byte的值。格式%#q将尽可能的使用反引号。（格式%q还用于整数和符文，产生一个带单引号的符文常量。）还有，%x用于字符串，字节数组和字节切片，以及整数，生成一个长的十六进制字符串，并且如果在格式中有一个空格（% x），其将会在字节中插入空格。

  另一个方便的格式是%T，其可以打印出值的类型。
  ```go
  fmt.Printf("%T\n", timeZone)
  ```
  会打印出
  ```go
  map[string] int
  ```
  如果你想控制自定义类型的缺省格式，只需要对该类型定义一个签名为String() string的方法。对于我们的简单类型T，看起来可能是这样的。
  ```go
  func (t *T) String() string {
      return fmt.Sprintf("%d/%g/%q", t.a, t.b, t.c)
  }
  fmt.Printf("%v\n", t)
  ```
  会按照如下格式打印
  ```go
  7/-2.35/"abc\tdef"
  ```
  （如果你需要打印类型为T的值，同时需要指向T的指针，那么String的接收者必须为值类型的；这个例子使用了指针，是因为这对于结构体类型更加有效和符合语言习惯。更多信息参见下面的章节pointers vs. value receivers）

  我们的String方法可以调用Sprintf，是因为打印程序是完全可重入的，并且可以按这种方式进行包装。然而，对于这种方式，有一个重要的细节需要明白：不要将调用Sprintf的String方法构造成无穷递归。如果Sprintf调用尝试将接收者直接作为字符串进行打印，就会导致再次调用该方法，发生这样的情况。这是一个很常见的错误，正如这个例子所示。
  ```go
  type MyString string

  func (m MyString) String() string {
      return fmt.Sprintf("MyString=%s", m) // Error: will recur forever.
  }
  ```
  这也容易修改：将参数转换为没有方法函数的，基本的字符串类型。
  ```go
  type MyString string
  func (m MyString) String() string {
      return fmt.Sprintf("MyString=%s", string(m)) // OK: note conversion.
  }
  ```
  在初始化章节，我们将会看到另一种避免该递归的技术。

  另一种打印技术，是将一个打印程序的参数直接传递给另一个这样的程序。Printf的签名使用了类型...interface{}作为最后一个参数，来指定在格式之后可以出现任意数目的（任意类型的）参数。
  ```go
  func Printf(format string, v ...interface{}) (n int, err error) { ... }
  ```
  在函数Printf内部，v就像是一个类型为[]interface{}的变量，但是如果其被传递给另一个可变参数的函数，其就像是一个正常的参数列表。这里有一个对我们上面用到的函数log.Println的实现。其将参数直接传递给fmt.Sprintln来做实际的格式化。
  ```go
  // Println prints to the standard logger in the manner of fmt.Println.
  func Println(v ...interface{}) {
      std.Output(2, fmt.Sprintln(v...))  // Output takes parameters (int, string)
  }
  ```
  我们在嵌套调用Sprintln中v的后面使用了...来告诉编译器将v作为一个参数列表；否则，其会只将v作为单个切片参数进行传递。

  除了我们这里讲到的之外，还有很多有关打印的技术。详情参见godoc文档中对fmt的介绍。

  顺便说下，...参数可以为一个特定的类型，例如...int，可以用于最小值函数，来选择整数列表中的最小值：
  ```go
  func Min(a ...int) int {
      min := int(^uint(0) >> 1)  // largest int
      for _, i := range a {
          if i < min {
              min = i
          }
      }
      return min
  }
  ```
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
有时候是需要分配一个二维切片的，例如这种情况可见于当扫描像素行的时候。有两种方式可以实现。一种是独立的分配每一个切片；另一种是分配单个数组，为其 指定单独的切片们。使用哪一种方式取决于你的应用。如果切片们可能会增大或者缩小，则它们应该被单独的分配以避免覆写了下一行；如果不会，则构建单个分配 会更加有效。作为参考，这里有两种方式的框架。首先是一次一行：
```go
// Allocate the top-level slice.
picture := make([][]uint8, YSize) // One row per unit of y.
// Loop over the rows, allocating the slice for each row.
for i := range picture {
	picture[i] = make([]uint8, XSize)
}
```
然后是分配一次，被切片成多行：
```go
// Allocate the top-level slice, the same as before.
picture := make([][]uint8, YSize) // One row per unit of y.
// Allocate one large slice to hold all the pixels.
pixels := make([]uint8, XSize*YSize) // Has type []uint8 even though picture is [][]uint8.
// Loop over the rows, slicing each row from the front of the remaining pixels slice.
for i := range picture {
	picture[i], pixels = pixels[:XSize], pixels[XSize:]
}
```
