module Lf = Loop_ir_test.Loop_fixtures
module Pcg = Walk_core.Pcg

type case = {
  seed : int64;
  kernel : Kernel.t;
  outputs : Tensor_id.t list;
  input : Tensor.packed;
}

(* Draws are [Pcg.next]'s [0, 2^32) as [int64], narrowed only after the
   remainder by a bound far below 2^30, so every runtime holds them. *)
type g = { mutable pcg : Pcg.t }

let below g bound =
  let x, pcg = Pcg.next g.pcg in
  g.pcg <- pcg;
  Int64.to_int (Int64.rem x (Int64.of_int bound))

let cells = 8
let shape = Lf.shape_w cells

let load id =
  Expr.Value.load
    (Expr_bridge.source_of_id (Lf.tid id))
    (Expr_bridge.coord_of_vec6 Symbolic.out_vec)

let constants = [| 0.5; 1.; 1.5; 2.; 0.1; 3. |]

(* The body of value [id], over the values and the input before it. *)
let body g id =
  let ia = below g id in
  let ib = below g id in
  let a = load ia and b = load ib in
  let k = Expr.Value.const constants.(below g (Array.length constants)) in
  let op = below g 6 in
  let value =
    match op with
    | 0 -> Expr.Value.add a k
    | 1 -> Expr.Value.mul a k
    | 2 -> Expr.Value.sub a b
    | 3 -> Expr.Value.add a b
    | 4 -> Expr.Value.mul a b
    | _ -> Expr.Value.sub k a
  in
  (* the values it reads: [b] only for the two-operand forms *)
  (value, if op = 0 || op = 1 || op = 5 then [ ia ] else [ ia; ib ])

let generate ~seed ~length =
  let g = { pcg = Pcg.seed ~seed ~seq:0x91acedL } in
  let read = Array.make (length + 1) false in
  let value id =
    let e, sources = body g id in
    List.iter (fun i -> read.(i) <- true) sources;
    {
      Kernel.Value.id = Lf.tid id;
      sg = Lf.sg id shape Lf.f32;
      computation = Region_group.Ref.Solo (Region_program.pixel e);
      result = Kernel.Result_conversion.Round_f32;
    }
  in
  let values = List.init length (fun i -> value (i + 1)) in
  (* a value nothing reads is an output; any other is one half the time *)
  let outputs =
    List.filter_map
      (fun i ->
        if i = length || (not read.(i)) || below g 2 = 0 then Some (Lf.tid i)
        else None)
      (List.init length (fun i -> i + 1))
  in
  let kernel =
    Err.or_raise ~pp_error:Kernel.pp_error
      (Kernel.create
         ~inputs:
           [
             {
               Kernel.Input.id = Lf.tid 0;
               sg = Lf.sg 0 shape Lf.f32;
               binding = Kernel.Binding.Caller;
             };
           ]
         ~values ~outputs ())
  in
  let input =
    Lf.f32_tensor shape (fun c ->
        let i = (Vec6.offset shape c :> int) in
        (float_of_int (i * 5 mod 11) *. 0.37) -. 1.)
  in
  { seed; kernel; outputs; input }

let bind c id = if Tensor_id.equal id (Lf.tid 0) then Some c.input else None
