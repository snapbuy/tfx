# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E2E Tests for tfx.examples.penguin.penguin_pipeline_local."""

import os
import unittest

from absl import logging
import tensorflow as tf

from tfx.dsl.io import fileio
from tfx.examples.penguin import penguin_pipeline_local
from tfx.examples.penguin import penguin_pipeline_local_e2e_test
from tfx.orchestration import metadata
from tfx.orchestration.local.local_dag_runner import LocalDagRunner


@unittest.skipIf(tf.__version__ < '2',
                 'Uses keras Model only compatible with TF 2.x')
class PenguinPipelineLocalFlaxEndToEndTest(
    penguin_pipeline_local_e2e_test.PenguinPipelineLocalEndToEndTest):

  def setUp(self):
    super(PenguinPipelineLocalFlaxEndToEndTest, self).setUp()

    try:
      import flax  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise unittest.SkipTest('flax is not installed. Skipping test.')

    self._module_file = os.path.join(
        os.path.dirname(__file__), 'penguin_utils_flax.py')

  def testPenguinPipelineLocal(self):
    pipeline = penguin_pipeline_local._create_pipeline(
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        enable_tuning=False,
        beam_pipeline_args=[])

    logging.info('Starting the first pipeline run.')
    LocalDagRunner().run(pipeline)

    self.assertTrue(fileio.exists(self._serving_model_dir))
    self.assertTrue(fileio.exists(self._metadata_path))
    expected_execution_count = 9  # 8 components + 1 resolver
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertEqual(expected_execution_count, execution_count)

    self.assertPipelineExecution(False)

    logging.info('Starting the second pipeline run. All components except '
                 'Evaluator and Pusher will use cached results.')
    LocalDagRunner().run(pipeline)

    with metadata.Metadata(metadata_config) as m:
      # Artifact count is increased by 3 caused by Evaluator and Pusher.
      self.assertLen(m.store.get_artifacts(), artifact_count + 3)
      artifact_count = len(m.store.get_artifacts())
      self.assertLen(m.store.get_executions(), expected_execution_count * 2)

    logging.info('Starting the third pipeline run. '
                 'All components will use cached results.')
    LocalDagRunner().run(pipeline)

    # Asserts cache execution.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is unchanged.
      self.assertLen(m.store.get_artifacts(), artifact_count)
      self.assertLen(m.store.get_executions(), expected_execution_count * 3)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
