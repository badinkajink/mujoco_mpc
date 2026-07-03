# Apply PATCH_FILE to TARGET_FILE in place with `patch -f`, TOLERATING failure.
# Used by copy_model_resources for the magpie gripper patch: a hunk reject must
# not abort the whole asset-staging target (a hard dep of libmjpc/agent_server) --
# worst case the Lean/reach tasks miss the gripper pads while stabilize/deploy
# (derived from the unpatched base BEFORE this step) is unaffected.
#
# This replaces `COMMAND bash -c "patch ... || true"`, which CMake 3.22's Ninja
# generator mangles (the `|| true` leaks out of the -c string, so the guard
# becomes a separate shell command and a patch failure kills the build).
execute_process(
  COMMAND patch -f --no-backup-if-mismatch ${TARGET_FILE} -i ${PATCH_FILE}
  RESULT_VARIABLE _patch_rc)
if(NOT _patch_rc EQUAL 0)
  message(WARNING
    "magpie patch did not fully apply (patch exit ${_patch_rc}); Lean/reach may "
    "be missing gripper bodies/pads -- stabilize/deploy models are unaffected. "
    "Regenerate ${PATCH_FILE} against humanoid_bench/h1_2_base/h1_2_pos.xml.")
endif()
