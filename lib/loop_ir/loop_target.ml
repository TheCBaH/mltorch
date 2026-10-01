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
  lanes : int;
  inner_loops : bool;
  support : Op.t -> support;
  cost : Op.t -> float;
}

(* The lanes of one binary64 register, and how many registers a logical vector
   spans. *)
let f64_lanes bits = bits / 64

(* What one lane costs scalar, in units of one binary64 add. A conversion
   narrows and widens (two instructions); a division or square root is slower
   than an add; a transcendental is a call; a bool store is a compare and a
   conditional. *)
let scalar_cost = function
  | Op.Add | Op.Sub | Op.Mul | Op.Neg_abs | Op.Logic | Op.Const | Op.Select
  | Op.Float_max | Op.Value_compare | Op.Splat | Op.Index_value ->
      1.
  | Op.Div | Op.Sqrt_trunc | Op.Round_f32 | Op.Bool_store -> 2.
  | Op.Broadcast_load | Op.Contiguous_load | Op.Convert_i32_load
  | Op.Strided_load ->
      1.
  | Op.Contiguous_store | Op.Strided_store -> 1.
  | Op.Transcendental -> 8.

(* Extract a lane, operate on it, put the result back: the overhead an expanded
   lane pays on top of its scalar operation. *)
let expansion_overhead = 1.

let make ?(inner_loops = true) ~name ~vector_bits ~lanes ~native ~native_cost ()
    =
  let support op = if native op then Native else Expanded in
  let cost op =
    match support op with
    | Native -> native_cost op
    | Expanded -> float_of_int lanes *. (scalar_cost op +. expansion_overhead)
  in
  { name; vector_bits; lanes; inner_loops; support; cost }

(* A native operation is one instruction per register a logical vector spans,
   at the unit cost listed (a conversion is two instructions, a division or
   square root slower than an add). *)
let native_cost ~vector_bits ~lanes op =
  let regs = float_of_int (lanes / f64_lanes vector_bits) in
  match op with
  | Op.Add | Op.Sub | Op.Mul | Op.Neg_abs | Op.Logic | Op.Const | Op.Select
  | Op.Float_max | Op.Value_compare ->
      regs
  | Op.Div | Op.Sqrt_trunc -> 2. *. regs
  | Op.Round_f32 -> 2. *. regs
  | Op.Contiguous_load | Op.Convert_i32_load | Op.Contiguous_store -> 2. *. regs
  | Op.Broadcast_load | Op.Splat | Op.Index_value -> regs +. 1.
  | Op.Bool_store | Op.Strided_load | Op.Strided_store | Op.Transcendental ->
      (* never native *)
      scalar_cost op

let native = function
  | Op.Bool_store | Op.Strided_load | Op.Strided_store | Op.Transcendental ->
      false
  | _ -> true

let wasm128 =
  let vector_bits = 128 and lanes = 4 in
  make ~name:"wasm128" ~vector_bits ~lanes ~native
    ~native_cost:(native_cost ~vector_bits ~lanes)
    ()

let neon128 =
  let vector_bits = 128 and lanes = 4 in
  make ~inner_loops:false ~name:"neon128" ~vector_bits ~lanes ~native
    ~native_cost:(native_cost ~vector_bits ~lanes)
    ()

let scalar =
  {
    name = "scalar";
    vector_bits = 64;
    lanes = 1;
    inner_loops = false;
    support = (fun _ -> Expanded);
    cost = (fun op -> scalar_cost op +. expansion_overhead);
  }

let forced t = { t with name = t.name ^ "+forced"; cost = (fun _ -> 0.) }
let all = [ neon128; scalar; wasm128 ]

let profitable t body =
  let vector =
    List.fold_left
      (fun acc (op, n) -> acc +. (float_of_int n *. t.cost op))
      0. body
  in
  let scalar =
    List.fold_left
      (fun acc (op, n) -> acc +. (float_of_int n *. scalar_cost op))
      0. body
  in
  vector < float_of_int t.lanes *. scalar
