(* Exercises the lib/core error framework against lib/native's local error sets:
   deterministic payload messages (golden), composition by row-union + printer
   delegation, automatic backtrace capture (structural, never golden — line
   numbers shift), and let* short-circuit + row unification. *)

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

let pp_dim_result pp_ok = Core.Pretty.core_result ~ok:pp_ok ~error:Dim.pp_error

(* Deterministic payload messages — the real regression guard. *)
let%expect_test "component pp_error messages" =
  Format.printf "%a@." Dim.pp_error (`Non_positive_extent (-3));
  Format.printf "%a@." Aten_shape.pp_error
    (`Rank_out_of_range { Aten_shape.rank = 7; lo = 0; hi = 6 });
  Format.printf "%a@." Symint.pp_error
    (`Out_of_range { Symint.name = "s0"; value = 17; lo = 1; hi = 1024 });
  Format.printf "%a@." Symint.pp_error
    (`Not_divisible { Symint.name = "s0"; value = 17; divisor = 8 });
  [%expect
    {|
    extent must be >= 1, got -3
    rank 7 out of [0, 6]
    symint s0 = 17 out of [1, 1024)
    symint s0 = 17 not divisible by 8 |}]

(* Extents are sizes, valid from 1: zero is rejected too, not only negatives —
   the engine has no empty tensors. *)
let%expect_test "extent floor is 1 (zero rejected)" =
  (match Dim.extent 0 with
  | _ -> print_string "no error\n"
  | exception Invalid_argument msg -> Printf.printf "raised: %s\n" msg);
  (match Dim.extent_checked 0 with
  | Ok _ -> print_string "ok\n"
  | Error e -> Format.printf "%a@." Dim.pp_error e.Core.Error.kind);
  [%expect
    {|
    raised: Dim.extent: must be >= 1
    extent must be >= 1, got 0 |}]

(* Composition: Aten_shape.error unions Dim.error, and its printer delegates the
   [#Dim.error] tag back to Dim.pp_error. Both values are typed [Aten_shape.error]. *)
let%expect_test "Aten_shape.pp_error delegates Dim tags to Dim.pp_error" =
  let own : Aten_shape.error =
    `Rank_out_of_range { Aten_shape.rank = 9; lo = 0; hi = 6 }
  in
  let from_dim : Aten_shape.error = `Non_positive_extent (-2) in
  Format.printf "%a@." Aten_shape.pp_error own;
  Format.printf "%a@." Aten_shape.pp_error from_dim;
  [%expect {|
    rank 9 out of [0, 6]
    extent must be >= 1, got -2 |}]

(* [fail] captures a stack automatically; [Core.Error.pp] renders message +
   provenance. Assert structure, not backtrace content (it shifts per build). *)
let%expect_test "fail captures backtrace; pp renders message + 'detected at:'" =
  match Dim.extent_checked (-3) with
  | Ok _ -> Format.printf "unexpected Ok@."
  | Error e ->
      assert (Printexc.raw_backtrace_length e.Core.Error.backtrace > 0);
      let s = Core.Pretty.to_string (Core.Error.pp Dim.pp_error) e in
      Format.printf "msg=%b provenance=%b@."
        (contains s "extent must be >= 1, got -3")
        (contains s "detected at:");
      [%expect {| msg=true provenance=true |}]

(* [of_option] is the option-shaped [fail]: the point of it over
   [Stdlib.Option.to_result] is that the failure carries a [Core.Error.t], so a
   bridged absence has provenance like any other. *)
let%expect_test "of_option bridges an option into the framework" =
  let pp ppf (`Missing k) = Format.fprintf ppf "no binding for %s" k in
  let show o =
    Format.printf "%a@."
      (Core.Pretty.core_result ~ok:Format.pp_print_int ~error:pp)
      (Core.of_option (`Missing "k") o)
  in
  show (Some 7);
  show None;
  [%expect {|
    7
    no binding for k |}]

(* The reason [of_option] is a [Core] helper and not [Option.to_result]: the
   CALLER must stay visible in the backtrace. A version that built the [Error.t]
   by hand with an empty backtrace would satisfy every assertion above and lose
   exactly this.

   Asserts that the calling file is named, not a frame index. [of_option] adds
   its own frame between [make] and the caller, and inlining decides which of
   those survives — on 4.14.3 [fail] inlines into [of_option] while [of_option]
   itself does not. Naming an index here would encode one compiler's choices. *)
let%expect_test "of_option keeps the caller in the backtrace" =
  let pp ppf (`Missing k) = Format.fprintf ppf "no binding for %s" k in
  match Core.of_option (`Missing "k") None with
  | Ok () -> Format.printf "unexpected Ok@."
  | Error e ->
      let s = Core.Pretty.to_string (Core.Error.pp pp) e in
      Format.printf "captured=%b caller_named=%b@."
        (Printexc.raw_backtrace_length e.Core.Error.backtrace > 0)
        (contains s "core_test.ml");
      [%expect {| captured=true caller_named=true |}]

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
    let* () = Core.fail (`Non_positive_extent (-1)) in
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
    extent must be >= 1, got -1 |}]

(* The remaining one-line combinators: >>|, >>=, map_error (lifts the payload,
   keeps the backtrace), and Core.List.map (short-circuits on first Error). *)
let%expect_test "combinators: >>| >>= map_error Core.List.map" =
  let open Core.Syntax in
  let r1 =
    (Core.return 4 >>| fun x -> x + 1) >>= fun x -> Core.return (x * 2)
  in
  let lifted =
    Core.fail (`Non_positive_extent (-5)) |> Core.map_error (fun _ -> `Renamed)
  in
  let pos x = if x >= 0 then Core.return x else Core.fail (`Neg x) in
  let summary =
    let* r1n = r1 in
    let lifted_name =
      match lifted with
      | Ok () -> "ok"
      | Error e -> if e.Core.Error.kind = `Renamed then "Renamed" else "?"
    in
    Core.return
      ( r1n,
        lifted_name,
        Core.List.map pos [ 1; 2; 3 ] = Ok [ 1; 2; 3 ],
        match Core.List.map pos [ 1; -2; 3 ] with
        | Error _ -> true
        | Ok _ -> false )
  in
  let pp_summary ppf (r1n, lifted, listok, listerr) =
    Format.fprintf ppf "r1=%d lifted=%s listok=%b listerr=%b" r1n lifted listok
      listerr
  in
  Format.printf "%a@." (pp_dim_result pp_summary) summary;
  [%expect {| r1=10 lifted=Renamed listok=true listerr=true |}]
