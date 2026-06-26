(* Exercises the lib/core error framework against lib/native's local error sets:
   deterministic payload messages (golden), composition by row-union + printer
   delegation, automatic backtrace capture (structural, never golden — line
   numbers shift), and let* short-circuit + row unification. *)

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

(* Deterministic payload messages — the real regression guard. *)
let%expect_test "component pp_error messages" =
  Format.printf "%a@." Dim.pp_error (`Negative_extent (-3));
  Format.printf "%a@." Aten_shape.pp_error
    (`Rank_out_of_range { Aten_shape.rank = 7; lo = 0; hi = 6 });
  Format.printf "%a@." Symint.pp_error
    (`Out_of_range { Symint.name = "s0"; value = 17; lo = 1; hi = 1024 });
  Format.printf "%a@." Symint.pp_error
    (`Not_divisible { Symint.name = "s0"; value = 17; divisor = 8 });
  [%expect
    {|
    extent must be >= 0, got -3
    rank 7 out of [0, 6]
    symint s0 = 17 out of [1, 1024)
    symint s0 = 17 not divisible by 8 |}]

(* Composition: Aten_shape.error unions Dim.error, and its printer delegates the
   [#Dim.error] tag back to Dim.pp_error. Both values are typed [Aten_shape.error]. *)
let%expect_test "Aten_shape.pp_error delegates Dim tags to Dim.pp_error" =
  let own : Aten_shape.error =
    `Rank_out_of_range { Aten_shape.rank = 9; lo = 0; hi = 6 }
  in
  let from_dim : Aten_shape.error = `Negative_extent (-2) in
  Format.printf "%a@." Aten_shape.pp_error own;
  Format.printf "%a@." Aten_shape.pp_error from_dim;
  [%expect {|
    rank 9 out of [0, 6]
    extent must be >= 0, got -2 |}]

(* [fail] captures a stack automatically; [Core.Error.pp] renders message +
   provenance. Assert structure, not backtrace content (it shifts per build). *)
let%expect_test "fail captures backtrace; pp renders message + 'detected at:'" =
  match Dim.extent_checked (-3) with
  | Ok _ -> Format.printf "unexpected Ok@."
  | Error e ->
      assert (Printexc.raw_backtrace_length e.Core.Error.backtrace > 0);
      let s = Format.asprintf "%a" (Core.Error.pp Dim.pp_error) e in
      Format.printf "msg=%b provenance=%b@."
        (contains s "extent must be >= 0, got -3")
        (contains s "detected at:");
      [%expect {| msg=true provenance=true |}]

(* let* stops at the first Error; the row unifies to the union of every tag the
   chain can raise (it only type-checks because Aten_shape.error holds both). *)
let%expect_test "let* short-circuits and unifies the error row" =
  let open Core.Syntax in
  let chain ~rank_bad : (int, Aten_shape.error) Core.result =
    let* () =
      if rank_bad then
        Core.fail (`Rank_out_of_range { Aten_shape.rank = 7; lo = 0; hi = 6 })
      else Core.return ()
    in
    let* () = Core.fail (`Negative_extent (-1)) in
    Core.return 99 (* unreached *)
  in
  let show ~rank_bad =
    match chain ~rank_bad with
    | Ok n -> Format.printf "ok %d@." n
    | Error e -> Format.printf "%a@." Aten_shape.pp_error e.Core.Error.kind
  in
  show ~rank_bad:true;
  show ~rank_bad:false;
  [%expect {|
    rank 7 out of [0, 6]
    extent must be >= 0, got -1 |}]

(* The remaining one-line combinators: >>|, >>=, map_error (lifts the payload,
   keeps the backtrace), and Core.List.map (short-circuits on first Error). *)
let%expect_test "combinators: >>| >>= map_error Core.List.map" =
  let open Core.Syntax in
  let r1 =
    (Core.return 4 >>| fun x -> x + 1) >>= fun x -> Core.return (x * 2)
  in
  let lifted =
    Core.fail (`Negative_extent (-5)) |> Core.map_error (fun _ -> `Renamed)
  in
  let pos x = if x >= 0 then Core.return x else Core.fail (`Neg x) in
  Format.printf "r1=%d lifted=%s listok=%b listerr=%b@." (Result.get_ok r1)
    (match lifted with
    | Ok () -> "ok"
    | Error e -> if e.Core.Error.kind = `Renamed then "Renamed" else "?")
    (Core.List.map pos [ 1; 2; 3 ] = Ok [ 1; 2; 3 ])
    (match Core.List.map pos [ 1; -2; 3 ] with
    | Error _ -> true
    | Ok _ -> false);
  [%expect {| r1=10 lifted=Renamed listok=true listerr=true |}]
