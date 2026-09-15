import argparse

from utils.tools import get_configs_of
from preprocessor.preprocessor import Preprocessor


#python preprocess.py --dataset DailyTalk
if __name__ == "__main__":
    #创建一个命令行参数解析器 parser，后面通过它可以定义、解析用户在终端输入的参数。
    parser = argparse.ArgumentParser()

    """向 parser 注册一个名为 --dataset 的命令行选项。
       type=str：这个参数的值要被解析成字符串。
       required=True：使用脚本时 必须 提供这个参数，否则会报错并打印帮助信息。
       help="name of dataset"：在你使用 -h/--help 时，会显示这段说明。"""
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    #真正去读取并解析命令行参数，把结果放到 args 对象里。
    args = parser.parse_args()
    #那三个yaml文件，args.dataset="DailyTalk"
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    preprocessor = Preprocessor(preprocess_config, model_config, train_config)
    preprocessor.build_from_path()
