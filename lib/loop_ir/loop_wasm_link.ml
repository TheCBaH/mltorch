module I = Wasm.Instr
module R = Loop_wasm_runtime

let ( @ ) a b = List.rev_append (List.rev a) b
let map f l = List.rev (List.rev_map f l)

(* The callees a kernel reached, closed under what the helpers themselves call,
   in [Callee.all] order. *)
let reached used =
  let rec close acc = function
    | [] -> acc
    | c :: rest ->
        if List.mem c acc then close acc rest
        else close (c :: acc) (R.Callee.deps c @ rest)
  in
  let set = close [] used in
  List.filter (fun c -> List.mem c set) R.Callee.all

let kernel_call k = I.Call (List.length R.Callee.all + k)

let link ~callees kernels =
  let callees = reached callees in
  let imports =
    List.filter (fun c -> Option.is_some (R.Callee.import c)) callees
  in
  let defined =
    List.filter (fun c -> Option.is_none (R.Callee.import c)) callees
  in
  let n_imports = List.length imports in
  let position c l =
    let rec go i = function
      | [] -> invalid_arg "Loop_wasm_link.link: callee not reached"
      | c' :: rest -> if c = c' then i else go (i + 1) rest
    in
    go 0 l
  in
  let helpers = List.length defined in
  let n_all = List.length R.Callee.all in
  let final pseudo =
    if pseudo >= n_all then n_imports + helpers + (pseudo - n_all)
    else
      let c = R.Callee.of_index pseudo in
      if Option.is_some (R.Callee.import c) then position c imports
      else n_imports + position c defined
  in
  let renumber (f : Wasm.Func.t) =
    { f with Wasm.Func.body = map (I.map_calls final) f.Wasm.Func.body }
  in
  let funcs =
    map (fun c -> renumber (Option.get (R.body c))) defined
    @ map renumber kernels
  in
  let imports =
    List.map
      (fun c ->
        {
          Wasm.Import.module_name = R.import_module;
          name = Option.get (R.Callee.import c);
          type_ = R.signature c;
        })
      imports
  in
  (imports, funcs, n_imports + helpers)

(* What a host should know before it compiles the module: the ABI, the
   post-MVP features it needs, what it imports, which helpers it carries and the
   numeric policy it implements. Deterministic text: it is a custom section of
   the module, so the module's digest (its identity) covers it. *)
let manifest ~callees (m : Wasm.Module.t) =
  let callees = reached callees in
  (* A list prints as [key: a b c]; an empty one as [key:], never with a
     trailing space. *)
  let line key items =
    key ^ ":" ^ String.concat "" (List.map (( ^ ) " ") items)
  in
  let names f = List.map R.Callee.name (List.filter f callees) in
  String.concat "\n"
    [
      "loop-wasm/1";
      line "features" (List.map Wasm_features.name (Wasm_features.of_module m));
      line "imports"
        (List.map
           (fun (i : Wasm.Import.t) ->
             i.Wasm.Import.module_name ^ "." ^ i.Wasm.Import.name)
           m.Wasm.Module.imports);
      line "helpers" (names (fun c -> Option.is_none (R.Callee.import c)));
      "numerics: working=f64 f32=round-and-widen fma=none reassociation=none \
       i64=modular index=i32-checked";
      "";
    ]
