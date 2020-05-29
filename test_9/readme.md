# 问题描述
+ 分布式训练
+ 在一个机子上，模拟两个cpu节点。实现分布式训练方式

# 要求
+ 在os.environ[‘TF_CONFIG’] 里配置’cluster’，写两个’worker’,由于在本机上模拟两个节点，所以节点名（ip）都写“localhost”，两个节点写不同的端口。
+ 使用两个脚本，distributed_mnist_node1.py和distributed_mnist_node2.py。两份脚本分别模拟各自在两个节点运行，故两个代码的’task’配置，一个是
{‘type’: ‘worker’, ‘index’: 0}，一个是’task’: {‘type’: ‘worker’, ‘index’: 1}。
+ 改好两份脚本后，分别开两个cmd 命令行窗口（ubuntu下开两个终端），在两个窗口分别运行：窗口1 “python distributed_mnist_node1.py”；窗口2 “python distributed_mnist_node2.py”

