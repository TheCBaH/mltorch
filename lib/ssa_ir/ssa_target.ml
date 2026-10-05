(* What a backend can do with a vector, as data: legality and cost kept apart.
   The numbers are the Loop planner's, restated because this library sees no
   Loop type; the bridge suite checks every target, precision and operation
   against {!Loop_ir.Loop_target}. *)

module Op = struct
  type t =
    | Add
    | Bool_store
    | Broadcast_load
    | Const
    | Contiguous_load
    | Contiguous_store
    | Convert_i32_load
    | Div
    | Float_max
    | Fma
    | Index_value
    | Logic
    | Neg_abs
    | Round_f32
    | Select
    | Splat
    | Sqrt_trunc
    | Strided_load
    | Strided_store
    | Sub
    | Mul
    | Transcendental
    | Value_compare

  let all =
    [
      Add;
      Bool_store;
      Broadcast_load;
      Const;
      Contiguous_load;
      Contiguous_store;
      Convert_i32_load;
      Div;
      Float_max;
      Fma;
      Index_value;
      Logic;
      Neg_abs;
      Round_f32;
      Select;
      Splat;
      Sqrt_trunc;
      Strided_load;
      Strided_store;
      Sub;
      Mul;
      Transcendental;
      Value_compare;
    ]

  let name = function
    | Add -> "add"
    | Bool_store -> "bool_store"
    | Broadcast_load -> "broadcast_load"
    | Const -> "const"
    | Contiguous_load -> "contiguous_load"
    | Contiguous_store -> "contiguous_store"
    | Convert_i32_load -> "convert_i32_load"
    | Div -> "div"
    | Float_max -> "float_max"
    | Fma -> "fma"
    | Index_value -> "index_value"
    | Logic -> "logic"
    | Neg_abs -> "neg_abs"
    | Round_f32 -> "round_f32"
    | Select -> "select"
    | Splat -> "splat"
    | Sqrt_trunc -> "sqrt_trunc"
    | Strided_load -> "strided_load"
    | Strided_store -> "strided_store"
    | Sub -> "sub"
    | Mul -> "mul"
    | Transcendental -> "transcendental"
    | Value_compare -> "value_compare"
end

type support = Native | Expanded

type t = {
  name : string;
  vector_bits : int;
  precision : Ssa_numerics.Precision.t;
  lanes : Ssa_type.Lanes.t;
  inner_loops : bool;
  fma : bool;
  relaxed_madd : bool;
  row_block : int;
  support : Op.t -> support;
  cost : Op.t -> float;
  at : Ssa_numerics.Precision.t -> t;
}

(* The lanes of one register in the working precision, and how many registers a
   logical vector spans. *)
let elem_bits = function
  | Ssa_numerics.Precision.F32 -> 32
  | Ssa_numerics.Precision.F64 -> 64

let reg_lanes ~precision bits = bits / elem_bits precision

(* What one lane costs scalar, in units of one binary64 add. A conversion
   narrows and widens (two instructions); a division or square root is slower
   than an add; a transcendental is a call; a bool store is a compare and a
   conditional. *)
let scalar_cost ~precision = function
  | Op.Add | Op.Sub | Op.Mul | Op.Neg_abs | Op.Logic | Op.Const | Op.Select
  | Op.Float_max | Op.Value_compare | Op.Splat | Op.Index_value ->
      1.
  | Op.Fma ->
      (* a multiply and an add when there is no fused instruction *)
      2.
  | Op.Round_f32 -> (
      (* the identity in binary32: nothing is emitted *)
      match precision with
      | Ssa_numerics.Precision.F32 -> 0.
      | Ssa_numerics.Precision.F64 -> 2.)
  | Op.Div | Op.Sqrt_trunc | Op.Bool_store -> 2.
  | Op.Broadcast_load | Op.Contiguous_load | Op.Convert_i32_load
  | Op.Strided_load ->
      1.
  | Op.Contiguous_store | Op.Strided_store -> 1.
  | Op.Transcendental -> 8.

(* Extract a lane, operate on it, put the result back: the overhead an expanded
   lane pays on top of its scalar operation. *)
let expansion_overhead = 1.

(* The logical lanes the vectorizer plans for: four binary64 lanes (two 128-bit
   registers) and, in binary32, [f32_lanes] = 16 (four registers). Sixteen
   independent output cells per block is what hides the latency of each cell's
   own sequential sum: measured on native NEON, eight lanes gave 1.5x over
   scalar binary64 on a dense matvec and sixteen 2.9x (see the fp32 design
   record). *)
let rec make ?(inner_loops = fun _ -> true) ?(fma = false)
    ?(relaxed_madd = false) ?(row_block = 1) ~name ~vector_bits ~f64_lanes
    ~f32_lanes ~native ~native_cost precision =
  let lanes =
    match precision with
    | Ssa_numerics.Precision.F32 -> f32_lanes
    | Ssa_numerics.Precision.F64 -> f64_lanes
  in
  let support op = if native op then Native else Expanded in
  let cost op =
    match support op with
    | Native -> native_cost ~precision ~lanes op
    | Expanded ->
        float_of_int lanes *. (scalar_cost ~precision op +. expansion_overhead)
  in
  let name =
    match precision with
    | Ssa_numerics.Precision.F32 -> name ^ "+f32"
    | Ssa_numerics.Precision.F64 -> name
  in
  {
    name;
    vector_bits;
    precision;
    lanes = Ssa_type.Lanes.of_int lanes;
    inner_loops = inner_loops precision;
    fma;
    relaxed_madd;
    row_block;
    support;
    cost;
    at =
      (fun p ->
        make ~inner_loops ~fma ~relaxed_madd ~row_block
          ~name:
            (match precision with
            | Ssa_numerics.Precision.F32 ->
                String.sub name 0 (String.length name - 4)
            | Ssa_numerics.Precision.F64 -> name)
          ~vector_bits ~f64_lanes ~f32_lanes ~native ~native_cost p);
  }

(* A native operation is one instruction per register a logical vector spans,
   at the unit cost listed (a conversion is two instructions, a division or
   square root slower than an add; a binary32 load or store converts nothing). *)
let native_cost ~vector_bits ~precision ~lanes op =
  let regs = float_of_int (lanes / reg_lanes ~precision vector_bits) in
  let convert =
    match precision with
    | Ssa_numerics.Precision.F32 -> 1.
    | Ssa_numerics.Precision.F64 -> 2.
  in
  match op with
  | Op.Add | Op.Sub | Op.Mul | Op.Neg_abs | Op.Logic | Op.Const | Op.Select
  | Op.Float_max | Op.Value_compare | Op.Fma ->
      regs
  | Op.Div | Op.Sqrt_trunc -> 2. *. regs
  | Op.Round_f32 -> (
      match precision with
      | Ssa_numerics.Precision.F32 -> 0.
      | Ssa_numerics.Precision.F64 -> 2. *. regs)
  | Op.Contiguous_load | Op.Convert_i32_load | Op.Contiguous_store ->
      convert *. regs
  | Op.Broadcast_load | Op.Splat | Op.Index_value -> regs +. 1.
  | Op.Bool_store | Op.Strided_load | Op.Strided_store | Op.Transcendental ->
      (* never native *)
      scalar_cost ~precision op

let native = function
  | Op.Bool_store | Op.Strided_load | Op.Strided_store | Op.Transcendental ->
      false
  | _ -> true

let wasm128 =
  let vector_bits = 128 in
  make ~name:"wasm128" ~vector_bits ~f64_lanes:4 ~f32_lanes:16 ~native
    ~native_cost:(fun ~precision ~lanes op ->
      native_cost ~vector_bits ~precision ~lanes op)
    Ssa_numerics.Precision.F64

let wasm128_relaxed =
  let base = wasm128 in
  let rec relaxed (t : t) =
    {
      t with
      name = t.name ^ "+relaxed";
      relaxed_madd = true;
      at = (fun p -> relaxed (t.at p));
    }
  in
  relaxed base

let neon128 =
  let vector_bits = 128 in
  (* Nested vector loops lose on native binary64 (measured) and win in
     binary32, where sixteen independent cells hide each sum's latency. *)
  make
    ~inner_loops:(function
      | Ssa_numerics.Precision.F32 -> true | Ssa_numerics.Precision.F64 -> false)
    ~fma:true ~row_block:2 ~name:"neon128" ~vector_bits ~f64_lanes:4
    ~f32_lanes:16 ~native
    ~native_cost:(fun ~precision ~lanes op ->
      native_cost ~vector_bits ~precision ~lanes op)
    Ssa_numerics.Precision.F64

let rec scalar =
  {
    name = "scalar";
    vector_bits = 64;
    precision = Ssa_numerics.Precision.F64;
    lanes = Ssa_type.Lanes.of_int 1;
    inner_loops = false;
    fma = false;
    relaxed_madd = false;
    row_block = 1;
    support = (fun _ -> Expanded);
    cost =
      (fun op ->
        scalar_cost ~precision:Ssa_numerics.Precision.F64 op
        +. expansion_overhead);
    at = (fun _ -> scalar);
  }

let rec forced t =
  {
    t with
    name = t.name ^ "+forced";
    cost = (fun _ -> 0.);
    at = (fun p -> forced (t.at p));
  }

let rec with_inner_loops inner_loops t =
  { t with inner_loops; at = (fun p -> with_inner_loops inner_loops (t.at p)) }

let rec with_row_block row_block t =
  { t with row_block; at = (fun p -> with_row_block row_block (t.at p)) }

let f32 t = t.at Ssa_numerics.Precision.F32
let all = [ neon128; scalar; wasm128; wasm128_relaxed ]

let profitable t body =
  let vector =
    List.fold_left
      (fun acc (op, n) -> acc +. (float_of_int n *. t.cost op))
      0. body
  in
  let scalar =
    List.fold_left
      (fun acc (op, n) ->
        acc +. (float_of_int n *. scalar_cost ~precision:t.precision op))
      0. body
  in
  vector < float_of_int (Ssa_type.Lanes.to_int t.lanes) *. scalar
