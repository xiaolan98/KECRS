# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# hack to make sure -m transformer/generator works as expected
from .transformer_rec_new import TransformerRecGeneratorAgent as GeneratorAgent  # noqa: F401
