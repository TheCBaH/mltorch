(* The exact int64 wire form: a decimal string, so a value past 2^53 survives a
   JSON round trip unchanged, and a malformed or out-of-range record is
   rejected instead of rounded. Legacy float-only params (no "exact" member)
   stay legacy. *)

let decode s =
  match Jsont_bytesrw.decode_string Factory.Arange.params_jsont s with
  | Ok p -> Ok p
  | Error _ -> Error ()

let params ~exact =
  Printf.sprintf {|{"start":0,"stop":4,"step":1,"fmt":"i64"%s}|}
    (match exact with None -> "" | Some e -> {|,"exact":|} ^ e)

let show s =
  match decode s with
  | Error () -> print_endline "rejected"
  | Ok p -> (
      match p.Factory.Arange.exact with
      | None -> print_endline "legacy (no exact)"
      | Some e -> Printf.printf "exact %Ld %Ld %Ld\n" e.start e.stop e.step)

let%expect_test "exact bounds round trip as decimal strings past 2^53" =
  let e =
    {|{"start":"9007199254740993","stop":"9223372036854775807","step":"1"}|}
  in
  show (params ~exact:(Some e));
  let e_min = {|{"start":"-9223372036854775808","stop":"0","step":"1"}|} in
  show (params ~exact:(Some e_min));
  [%expect
    {|
    exact 9007199254740993 9223372036854775807 1
    exact -9223372036854775808 0 1 |}]

let%expect_test "an encoded exact record decodes to the same int64s" =
  let p =
    {
      Factory.Arange.start = 0.;
      stop = 4.;
      step = 1.;
      fmt = Payload.Fmt Payload.I64;
      exact =
        Some
          {
            Factory.Arange.Exact.start = 9_007_199_254_740_993L;
            stop = Int64.max_int;
            step = 1L;
          };
    }
  in
  (match Jsont_bytesrw.encode_string Factory.Arange.params_jsont p with
  | Error _ -> print_endline "encode failed"
  | Ok s ->
      print_endline s;
      show s);
  [%expect
    {|
    {"start":0,"stop":4,"step":1,"fmt":"i64","exact":{"start":"9007199254740993","stop":"9223372036854775807","step":"1"}}
    exact 9007199254740993 9223372036854775807 1 |}]

let%expect_test "malformed or out-of-range exact records are rejected" =
  List.iter
    (fun e -> show (params ~exact:(Some e)))
    [
      (* one past Int64.max_int *)
      {|{"start":"9223372036854775808","stop":"0","step":"1"}|};
      (* one below Int64.min_int *)
      {|{"start":"-9223372036854775809","stop":"0","step":"1"}|};
      (* not a decimal *)
      {|{"start":"12abc","stop":"0","step":"1"}|};
      {|{"start":"","stop":"0","step":"1"}|};
      (* a JSON number is exactly what the string form exists to avoid *)
      {|{"start":9007199254740993,"stop":0,"step":1}|};
      (* a missing member *)
      {|{"start":"0","stop":"4"}|};
    ];
  [%expect
    {|
    rejected
    rejected
    rejected
    rejected
    rejected
    rejected |}]

let%expect_test "params without an exact member stay legacy float" =
  show (params ~exact:None);
  [%expect {| legacy (no exact) |}]
