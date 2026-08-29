(* group_norm.default, and a pooling option native params cannot hold. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/group_norm_test.ml]. *)

open Helpers

(* --- group_norm.default -----------------------------------------------------

   NOT bound to a real ATen C symbol (see the [Op_bridge] arm's own comment on
   why), so no [Interp_dispatch]/[Interp_verify] layer here -- Op_bridge-only,
   with hand-computed values the same way [compute_test.ml]'s tests are, not
   leaning on ATen. *)

(* NCHW input [N=1,C=4,H=2,W=1], x[c,h] = h*10+c, so the flat NCHW data is
   [0,10, 1,11, 2,12, 3,13] -- group 0 is channels {0,1}, group 1 is {2,3},
   the same values [compute_test.ml]'s hand-computed group_norm test uses
   (there in native [H,W,C] order). No weight/bias: exercises the
   optional-absent fill path through the bridge and the NCHW<->NHWC relayout
   together. *)
let%expect_test
    "dispatch: group_norm.default splits channels into groups, no affine" =
  dispatch_print ~target:"torch.ops.aten.group_norm.default"
    ~bindings:
      [
        ( "input",
          float_tensor [ 1; 4; 2; 1 ] [ 0.; 10.; 1.; 11.; 2.; 12.; 3.; 13. ] );
      ]
    ~inputs:[ in_tensor "input"; in_int "num_groups" 2; in_float "eps" 0. ]
    ~noutputs:0;
  [%expect
    {| tensor f32 [H=4 W=2 C=1] {-1.09454, 0.895533, -0.895533, 1.09454, -1.09454, 0.895533, -0.895533, 1.09454} |}]

(* The per-channel affine, applied AFTER the relayout back to NCHW -- weight
   and bias are rank-1 [C] tensors, which right-align onto native C with no
   permute needed, unlike [input]. *)
let%expect_test "dispatch: group_norm.default applies weight/bias per channel" =
  dispatch_print ~target:"torch.ops.aten.group_norm.default"
    ~bindings:
      [
        ("input", float_tensor [ 1; 4; 1; 1 ] [ 0.; 1.; 2.; 3. ]);
        ("weight", float_tensor [ 4 ] [ 10.; 20.; 30.; 40. ]);
        ("bias", float_tensor [ 4 ] [ 100.; 200.; 300.; 400. ]);
      ]
    ~inputs:
      [
        in_tensor "input";
        in_int "num_groups" 2;
        in_tensor "weight";
        in_tensor "bias";
        in_float "eps" 0.;
      ]
    ~noutputs:0;
  [%expect {| tensor f32 [H=4 W=1 C=1] {90, 220, 270, 440} |}]

(* [num_groups] must be positive ([Op_config.Bad.pos], the bridge's own
   check) and must divide the channel count ([Norm.GroupNorm.output_shape]'s
   own check, not restated here). *)
let%expect_test "dispatch: group_norm.default rejects a bad num_groups" =
  dispatch_print ~target:"torch.ops.aten.group_norm.default"
    ~bindings:[ ("input", float_tensor [ 1; 4; 1; 1 ] [ 0.; 1.; 2.; 3. ]) ]
    ~inputs:[ in_tensor "input"; in_int "num_groups" 0 ]
    ~noutputs:0;
  dispatch_print ~target:"torch.ops.aten.group_norm.default"
    ~bindings:[ ("input", float_tensor [ 1; 4; 1; 1 ] [ 0.; 1.; 2.; 3. ]) ]
    ~inputs:[ in_tensor "input"; in_int "num_groups" 3 ]
    ~noutputs:0;
  [%expect
    {|
    error: group_norm.default: groups must be positive, got 0
    error: group_norm: channel count 4 is not divisible by num_groups 3 |}]

(* A TRANSPOSED convolution's ATen weight is [Cin, Cout/groups, kH, kW], so its
   output channel count -- and therefore its bias extent -- comes from
   [weight.C * groups], not from [weight.N] as it does for every other affine op.
   [Conv.Convolution.bias_shape] has always encoded that; the shared bias check
   briefly did not consult it, and rejected every valid transposed convolution
   whose input and output channel counts differ.

   Unequal channel counts are the whole point of the fixture: with Cin = Cout the
   two rules agree and the case is invisible. *)
let%expect_test "dispatch: transposed convolution bias uses weight.C * groups" =
  let x = float_tensor [ 1; 2; 2; 2 ] (List.init 8 float_of_int) in
  let w = float_tensor [ 2; 3; 1; 1 ] (List.init 6 float_of_int) in
  let b = float_tensor [ 3 ] [ 1.; 2.; 3. ] in
  dispatch_print ~target:"torch.ops.aten.convolution.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", b) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_tensor "bias";
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 0; 0 ];
        in_ints "dilation" [ 1; 1 ];
        in_bool "transposed" true;
        in_ints "output_padding" [ 0; 0 ];
        in_int "groups" 1;
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [H=3 W=2 C=2] {13, 16, 19, 22, 18, 23, 28, 33, ...} |}]

(* Ordinary malformed op-config arguments must be CONTAINED at this boundary --
   the bridge returns [Some (Error _)], never raises. [Dim.extent],
   [Op_config.Pos.of_int] and [Nonneg.of_int] all assert their preconditions,
   and the aggregate bounds added for Group 2 check magnitude only, so every one
   of these still reaches an assertion; what keeps it from escaping is that the
   construction happens inside the arm's exception boundary.

   These four are the cheapest cases that reach a different constructor each:
   groups zero (a zero channel PRODUCT, so [Dim.extent]), stride zero and
   dilation zero ([Pos]), negative padding ([Nonneg]). *)
(* The payload, not the rendering. The point of a structured row is that a
   caller can BRANCH on it -- which is what an undifferentiated
   [`Validation_failure "Op_config.Pos.of_int: not positive"] cannot support,
   and what these arms used to return. Destructuring here is the assertion that
   the fields exist and carry the op, the parameter and the offending value. *)
let config_fault ~target ~inputs bindings =
  let env = List.fold_left (fun m (k, t) -> Sm.add k t m) Sm.empty bindings in
  let node =
    PT.Node.make target inputs [ targ "out0" ] Sm.empty None (Some "test")
  in
  match Op_bridge.dispatch ~aten_env:env node with
  | None -> print_string "no native impl\n"
  | Some (Ok _) -> print_string "accepted\n"
  | Some (Error e) -> (
      match Err.Error.kind e with
      | `Bad_config { Op_config.Bad.op; param; fault } ->
          Format.printf "op=%s param=%a fault=%s@." op Op_config.Bad.pp_param
            param
            (match fault with
            | `Not_positive n -> Printf.sprintf "not_positive %d" n
            | `Negative n -> Printf.sprintf "negative %d" n)
      | other -> Format.printf "OTHER ROW: %a@." Op_bridge.pp_error other)

let%expect_test
    "dispatch: every Group-2 bridge arm reports a structured config fault" =
  let x = float_tensor [ 1; 2; 4; 4 ] (List.init 32 float_of_int) in
  let w = float_tensor [ 2; 2; 1; 1 ] (List.init 4 float_of_int) in
  let conv target extra =
    config_fault ~target
      ~inputs:([ in_tensor "input"; in_tensor "weight" ] @ extra)
      [ ("input", x); ("weight", w) ]
  in
  conv "torch.ops.aten.conv2d.default" [ in_ints "stride" [ 0; 1 ] ];
  conv "torch.ops.aten.conv2d.default" [ in_ints "padding" [ 0; -2 ] ];
  conv "torch.ops.aten.conv2d.padding"
    [ in_string "padding" "same"; in_ints "dilation" [ 3; 0 ] ];
  conv "torch.ops.aten.conv2d.padding"
    [ in_string "padding" "valid"; in_int "groups" 0 ];
  (* [padding] and [output_padding] are DIFFERENT arguments of the same op, so
     the op name cannot disambiguate them -- only the tag can. Both spellings
     appear here for that reason, and they must differ. *)
  conv "torch.ops.aten.convolution.default"
    [
      in_ints "padding" [ 0; -1 ];
      in_bool "transposed" false;
      in_ints "output_padding" [ 0; 0 ];
    ];
  conv "torch.ops.aten.convolution.default"
    [
      in_ints "padding" [ 0; 0 ];
      in_bool "transposed" true;
      in_ints "output_padding" [ 0; -1 ];
    ];
  (* Both COMPONENTS, not just one: they are validated by separate calls, and a
     tag fixed on only one of them looks correct from whichever side is
     tested. *)
  conv "torch.ops.aten.convolution.default"
    [
      in_ints "padding" [ 0; 0 ];
      in_bool "transposed" true;
      in_ints "output_padding" [ -1; 0 ];
    ];
  config_fault ~target:"torch.ops.aten.max_pool2d.default"
    ~inputs:[ in_tensor "self"; in_ints "kernel_size" [ 0; 2 ] ]
    [ ("self", x) ];
  config_fault ~target:"torch.ops.aten.max_pool2d.default"
    ~inputs:
      [
        in_tensor "self";
        in_ints "kernel_size" [ 2; 2 ];
        in_ints "padding" [ -1; 0 ];
      ]
    [ ("self", x) ];
  [%expect
    {|
    op=torch.ops.aten.conv2d.default param=stride fault=not_positive 0
    op=torch.ops.aten.conv2d.default param=padding fault=negative -2
    op=torch.ops.aten.conv2d.padding param=dilation fault=not_positive 0
    op=torch.ops.aten.conv2d.padding param=groups fault=not_positive 0
    op=torch.ops.aten.convolution.default param=padding fault=negative -1
    op=torch.ops.aten.convolution.default param=output_padding fault=negative -1
    op=torch.ops.aten.convolution.default param=output_padding fault=negative -1
    op=torch.ops.aten.max_pool2d.default param=kernel_size fault=not_positive 0
    op=torch.ops.aten.max_pool2d.default param=padding fault=negative -1 |}]

(* The padding MODE, which is model data and was reaching an asserting parser.
   Contained by the arm's [try], but as a [`Validation_failure] string -- where
   the serialized importer returns the offered mode in a typed row. Both now
   report the same thing, through one checked parser, so the accepted set cannot
   drift between them. Destructured, not just rendered. *)
let%expect_test "dispatch: an unsupported conv2d.padding mode is a typed row" =
  let x = float_tensor [ 1; 2; 4; 4 ] (List.init 32 float_of_int) in
  let w = float_tensor [ 2; 2; 1; 1 ] (List.init 4 float_of_int) in
  let mode m =
    let env = Sm.add "input" x (Sm.add "weight" w Sm.empty) in
    let node =
      PT.Node.make "torch.ops.aten.conv2d.padding"
        [ in_tensor "input"; in_tensor "weight"; in_string "padding" m ]
        [ targ "out0" ]
        Sm.empty None (Some "test")
    in
    match Op_bridge.dispatch ~aten_env:env node with
    | None -> print_string "no native impl\n"
    | Some (Ok _) -> Format.printf "%S accepted@." m
    | Some (Error e) -> (
        match Err.Error.kind e with
        | `Unsupported_padding_mode s -> Format.printf "mode=%S refused@." s
        | other -> Format.printf "OTHER ROW: %a@." Op_bridge.pp_error other)
  in
  mode "reflect";
  mode "SAME";
  mode "";
  mode "valid";
  mode "same";
  [%expect
    {|
    mode="reflect" refused
    mode="SAME" refused
    mode="" refused
    "valid" accepted
    "same" accepted |}]

let%expect_test "dispatch: malformed conv2d config is contained, not raised" =
  let x = float_tensor [ 1; 2; 2; 2 ] (List.init 8 float_of_int) in
  let w = float_tensor [ 2; 2; 1; 1 ] (List.init 4 float_of_int) in
  let conv extra =
    dispatch_print ~target:"torch.ops.aten.conv2d.default"
      ~bindings:[ ("input", x); ("weight", w) ]
      ~inputs:([ in_tensor "input"; in_tensor "weight" ] @ extra)
      ~noutputs:1
  in
  conv [ in_int "groups" 0 ];
  conv [ in_ints "stride" [ 0; 1 ] ];
  conv [ in_ints "padding" [ -1; 0 ] ];
  conv [ in_ints "dilation" [ 0; 1 ] ];
  (* and the well-formed node still lowers *)
  conv [ in_int "groups" 1 ];
  [%expect
    {|
    error: torch.ops.aten.conv2d.default: groups must be positive, got 0
    error: torch.ops.aten.conv2d.default: stride must be positive, got 0
    error: torch.ops.aten.conv2d.default: padding must not be negative, got -1
    error: torch.ops.aten.conv2d.default: dilation must be positive, got 0
    tensor f32 [H=2 W=2 C=2] {4, 5, 6, 7, 12, 17, 22, 27} |}]

(* RANK, which right-alignment into the six-axis frame erases. A bias declared
   [1,Cout] arrives indistinguishable from [Cout], so the shared [Graph_shape]
   check passes it -- while ATen refuses it ("expected bias to be
   1-dimensional"). The rank survives only on the ATen tensor. *)
let%expect_test "dispatch: a leading-singleton bias is refused on rank" =
  let x = float_tensor [ 1; 2; 2; 2 ] (List.init 8 float_of_int) in
  let w = float_tensor [ 3; 2; 1; 1 ] (List.init 6 float_of_int) in
  let conv_bias b =
    dispatch_print ~target:"torch.ops.aten.conv2d.default"
      ~bindings:[ ("input", x); ("weight", w); ("bias", b) ]
      ~inputs:[ in_tensor "input"; in_tensor "weight"; in_tensor "bias" ]
      ~noutputs:1
  in
  conv_bias (float_tensor [ 1; 3 ] [ 1.; 2.; 3. ]);
  conv_bias (float_tensor [ 3 ] [ 1.; 2.; 3. ]);
  [%expect
    {|
    error: bias must be rank-1, got rank-2
    tensor f32 [H=3 W=2 C=2] {5, 6, 7, 8, 14, 19, 24, 29, ...} |}]

(* ---- a pooling option the native params cannot hold, and one that they can *)

(* [dilation]: both arms decoded kernel/stride/padding and never read it.
   [Pool.MaxPool2d.params] has no field for it, so a serialized non-default
   value was not approximated -- it was DROPPED, and the bridge computed an
   undilated pool under the dilated name. [ceil_mode] DOES have a field (see
   [Pool.MaxPool2d.params.ceil_mode]) and is carried through, verified here by
   hand against the 5x5 input's actual max values (odd extent, so floor and
   ceil division disagree -- ATen's own real-oracle agreement is exercised
   separately by the generated `max_pool2d(_with_indices)` walks, not by this
   hand-derived fixture).

   The default spellings still pass, which is the half of this that a rejection
   test alone would not show: refusing every export that mentions the argument
   would be its own regression. *)
let%expect_test "dispatch: max_pool2d dilation is refused, ceil_mode is carried"
    =
  let a =
    float_tensor [ 1; 1; 5; 5 ] (List.init 25 (fun i -> float_of_int i))
  in
  let pool ?dilation ?ceil_mode target =
    dispatch_print ~target
      ~bindings:[ ("self", a) ]
      ~inputs:
        ([ in_tensor "self"; in_ints "kernel_size" [ 2; 2 ] ]
        @ (match dilation with
          | None -> []
          | Some d -> [ in_ints "dilation" d ])
        @
        match ceil_mode with
        | None -> []
        | Some b -> [ in_bool "ceil_mode" b ])
      ~noutputs:(if target = "torch.ops.aten.max_pool2d.default" then 1 else 2)
  in
  let both f =
    List.iter f
      [
        "torch.ops.aten.max_pool2d.default";
        "torch.ops.aten.max_pool2d_with_indices.default";
      ]
  in
  both (fun t -> pool ~dilation:[ 2; 2 ] t);
  both (fun t -> pool ~ceil_mode:true t);
  (* the defaults, written out, still lower *)
  both (fun t -> pool ~dilation:[ 1; 1 ] ~ceil_mode:false t);
  (* A single int is ATen's other legal spelling and normalizes to (1,1); three
     is not, and a value-only check accepted it because no element differed. *)
  both (fun t -> pool ~dilation:[ 1 ] t);
  both (fun t -> pool ~dilation:[ 1; 1; 1 ] t);
  [%expect
    {|
    error: torch.ops.aten.max_pool2d.default: dilation=[2, 2] is not supported (only 1)
    error: torch.ops.aten.max_pool2d_with_indices.default: dilation=[2, 2] is not supported (only 1)
    tensor f32 [W=3 C=3] {6, 8, 9, 16, 18, 19, 21, 23, ...}
    tensor f32 [W=3 C=3] {6, 8, 9, 16, 18, 19, 21, 23, ...}
    tensor f32 [W=2 C=2] {6, 8, 16, 18}
    tensor f32 [W=2 C=2] {6, 8, 16, 18}
    tensor f32 [W=2 C=2] {6, 8, 16, 18}
    tensor f32 [W=2 C=2] {6, 8, 16, 18}
    error: dilation: expected [h; w] or [v], got [1, 1, 1]
    error: dilation: expected [h; w] or [v], got [1, 1, 1] |}]
