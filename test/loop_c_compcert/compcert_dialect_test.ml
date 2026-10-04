open Loop_ir

let program = snd (List.hd Loop_vector_programs.all)

let%expect_test "the dialect refuses the vector form with a typed error" =
  (match
     Err.payload
       (Loop_c.kernel ~dialect:Loop_c_dialect.Compcert_scalar
          ~vector:Loop_target.neon128 ~name:"k" program)
   with
  | Ok _ -> print_endline "accepted"
  | Error e -> Fmt.pr "%a@." Loop_c.pp_error e);
  [%expect {| the Compcert_scalar dialect has no vector form |}]

let%expect_test "the dialect's prelude declares exactly the host symbols" =
  let prelude = Loop_c_runtime.prelude_in Loop_c_dialect.Compcert_scalar in
  let declared =
    List.filter_map
      (fun l ->
        if String.starts_with ~prefix:"extern " l then
          match String.index_opt l '(' with
          | Some i ->
              let head = String.sub l 0 i in
              let name = List.hd (List.rev (String.split_on_char ' ' head)) in
              Some (String.concat "" (String.split_on_char '*' name))
          | None -> None
        else None)
      (String.split_on_char '\n' prelude)
  in
  Fmt.pr "%b@." (List.sort compare declared = Loop_c_dialect.host_symbols);
  [%expect {| true |}]
