open Loop_ir
open Loop_fixtures

let emit = Loop_js.emit

let%expect_test "emitted JavaScript is a named function over typed arrays" =
  print_string (emit Loop_programs.doubling);
  [%expect
    {|
    function loop_kernel(b0, b1) {
      for (let i0 = 0; i0 < 4; i0++) {
        b1[(((((0 * 1 + 0) * 1 + 0) * 1 + 0) * 4 + i0) * 1 + 0)] = Math.fround((b0[(((((0 * 1 + 0) * 1 + 0) * 1 + 0) * 4 + i0) * 1 + 0)] * 2));
      }
      return null;
    } |}]

let%expect_test "emitting twice is byte-identical, whatever the allocation ids"
    =
  let build ~var ~temp:t =
    let i = Loop_index.Var (v var) in
    program
      ~buffers:[ buffer 1 (shape_w 2) f32 Loop_buffer.Output ]
      [
        Loop_stmt.For
          {
            var = v var;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const 2;
            body =
              [
                Loop_stmt.Assign
                  (Loop_carrier.Float, temp t, Loop_expr.Value_of_index i);
                Loop_stmt.Store
                  {
                    buffer = buffer 1 (shape_w 2) f32 Loop_buffer.Output;
                    coord = at_w i;
                    value =
                      Loop_stored.F32
                        (Loop_expr.Temp (Loop_carrier.Float, temp t));
                  };
              ];
          };
      ]
  in
  let a = emit (build ~var:0 ~temp:0) in
  Fmt.pr "%b@." (String.equal a (emit (build ~var:0 ~temp:0)));
  Fmt.pr "%b@." (String.equal a (emit (build ~var:31 ~temp:77)));
  [%expect {|
    true
    true |}]
