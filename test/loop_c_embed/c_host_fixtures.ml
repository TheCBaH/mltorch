open Loop_ir

(* A bundle whose invocation [at] fails unconditionally with [failure]. *)
let poisoned_bundle b ~at failure =
  let invocations =
    List.mapi
      (fun i (inv : Loop_bundle.invocation) ->
        if i <> at then inv
        else
          let p = inv.Loop_bundle.program in
          let always =
            Loop_bool.Index_eq (Loop_index.Const 0, Loop_index.Const 0)
          in
          {
            inv with
            Loop_bundle.program =
              {
                p with
                Loop_program.body =
                  Loop_stmt.Fail_if (always, failure) :: p.Loop_program.body;
              };
          })
      b.Loop_bundle.invocations
  in
  { b with Loop_bundle.invocations }
