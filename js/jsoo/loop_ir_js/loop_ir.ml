(* Recreates [lib/loop_ir]'s wrapped namespace as plain aliases over this
   (unwrapped) mirror's own flat modules -- see js/jsoo/loop_ir_js/dune's
   comment for why. Every module [lib/loop_ir] exposes, alphabetically. Add
   a new lib/loop_ir module here too, or [open Loop_ir] in a mirrored
   consumer (js/loop_js_exec/loop_js_exec.ml) silently stops seeing it. *)

module Loop_array = Loop_array
module Loop_bool = Loop_bool
module Loop_buffer = Loop_buffer
module Loop_carrier = Loop_carrier
module Loop_check = Loop_check
module Loop_expr = Loop_expr
module Loop_failure = Loop_failure
module Loop_index = Loop_index
module Loop_index_map = Loop_index_map
module Loop_interp = Loop_interp
module Loop_js = Loop_js
module Loop_js_failure = Loop_js_failure
module Loop_js_runtime = Loop_js_runtime
module Loop_linear = Loop_linear
module Loop_lower = Loop_lower
module Loop_lower_ctx = Loop_lower_ctx
module Loop_lower_index = Loop_lower_index
module Loop_lower_region = Loop_lower_region
module Loop_lower_value = Loop_lower_value
module Loop_mark = Loop_mark
module Loop_node_program = Loop_node_program
module Loop_opt = Loop_opt
module Loop_opt_collapse = Loop_opt_collapse
module Loop_opt_cse = Loop_opt_cse
module Loop_opt_fold = Loop_opt_fold
module Loop_opt_guards = Loop_opt_guards
module Loop_opt_hoist = Loop_opt_hoist
module Loop_opt_scope = Loop_opt_scope
module Loop_opt_simplify = Loop_opt_simplify
module Loop_opt_unit_loops = Loop_opt_unit_loops
module Loop_pp = Loop_pp
module Loop_program = Loop_program
module Loop_range = Loop_range
module Loop_region_program = Loop_region_program
module Loop_stmt = Loop_stmt
module Loop_stored = Loop_stored
module Loop_temp = Loop_temp
module Loop_unsupported = Loop_unsupported
module Loop_var = Loop_var
