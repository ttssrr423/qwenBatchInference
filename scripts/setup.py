from setuptools import setup, find_packages

setup (
    name = "liteqwen_py",
    version = "0.0.1",
    author = "tangshirui",
    author_email = "2040179500@qq.com",
    description = "liteqwen python api",
    # url = "",
    packages = ['liteqwen_py'],

    package_data = {
        '': ['*.dll', '*.so', '*.dylib']
    }
)