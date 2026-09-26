open Loop_ir

(* A representative elementwise kernel for the Loop IR benchmarks: one
   W-shaped buffer of [size] elements,
     out[w] = round_f32(sin(2*in[w]+1) * exp(0.001*in[w]) + in[w])
   Hand-lowered like test/loop_ir/loop_programs.ml's [doubling] -- no
   Kernel/Region/Fusion_plan involved, deliberately: the benchmark needs a
   buffer large enough that a warm run's cost is dominated by the loop body,
   not by call or compile overhead, which the tiny [Loop_fixtures] shapes
   (built for foundation tests, not timing) are not meant to give.

   Shared, unmodified, between the native benchmark (loop_bench_native.ml,
   this directory) and the jsoo one (js/loop_js_exec/bench, which copies this
   file rather than forking it), so every reported number describes the same
   program. *)

let shape_w n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:n ~c:1
let f32 = Payload.Fmt Payload.F32
let tid = Tensor_id.of_int
let sg id shape fmt = Tensor_sig.create ~id:(tid id) ~name:"" ~shape ~fmt ()

let buffer id shape fmt role =
  { Loop_buffer.id = tid id; sg = sg id shape fmt; role }

let at_w (i : Loop_index.t) =
  Expr.Coord.make ~n:(Loop_index.Const 0) ~t:(Loop_index.Const 0)
    ~d:(Loop_index.Const 0) ~h:(Loop_index.Const 0) ~w:i ~c:(Loop_index.Const 0)

let v n = Loop_var.of_int n
let size = 4096
let input = buffer 0 (shape_w size) f32 Loop_buffer.Input
let output = buffer 1 (shape_w size) f32 Loop_buffer.Output
let w = Loop_index.Var (v 0)

let expr =
  let x = Loop_expr.Load (input, at_w w) in
  let two_x_plus_1 =
    Loop_expr.Binary
      ( Expr.Value.Add,
        Loop_expr.Binary (Expr.Value.Mul, x, Loop_expr.Const 2.),
        Loop_expr.Const 1. )
  in
  let decay = Loop_expr.Binary (Expr.Value.Mul, x, Loop_expr.Const 0.001) in
  Loop_expr.Round_f32
    (Loop_expr.Binary
       ( Expr.Value.Add,
         Loop_expr.Binary
           ( Expr.Value.Mul,
             Loop_expr.Unary (Expr.Value.Sin, two_x_plus_1),
             Loop_expr.Unary (Expr.Value.Exp, decay) ),
         x ))

let program =
  {
    Loop_program.buffers = [ input; output ];
    body =
      [
        Loop_stmt.For
          {
            var = v 0;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const size;
            body =
              [
                Loop_stmt.Store
                  {
                    buffer = output;
                    coord = at_w w;
                    value = Loop_stored.F32 expr;
                  };
              ];
          };
      ];
    scan_limits = Expr.Scan_limits.default;
    max_depth = 64;
  }

let data = Array.init size (fun i -> (float_of_int (i mod 199) -. 99.) *. 0.25)

let bind id =
  if Tensor_id.equal id (tid 0) then
    Some
      (Tensor.materialize (shape_w size) (fun c ->
           data.(Dim.to_int (Vec6.get c Axis.W))))
  else None
