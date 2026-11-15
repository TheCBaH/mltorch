open Loop_wasm_ctx

type t = {
  module_ : Wasm.Module.t;
  local_base : Layout.Address.t;
  mark_base : Layout.Address.t option;
  heap_base : Layout.Address.t;
  sites : Loop_failure.t array;
  precision : Loop_numerics.Precision.t;
}

type kernel = {
  func : Wasm.Func.t;
  callees : R.Callee.t list;
  local_bytes : int64;
  data : Wasm.Data.t list;
  sites : Loop_failure.t array;
  precision : Loop_numerics.Precision.t;
  refusal : Loop_numerics.Refusal.t option;
}

let align16 x = Int64.logand (Int64.add x 15L) (Int64.lognot 15L)

let lower_with ?(numerics = Loop_numerics.Reference_f64) ?(count_marks = false)
    (produce :
      mark_base:Layout.Address.t option ->
      table_alloc:(bytes:int64 -> Layout.Address.t) ->
      ('a, 'e) Err.t) =
  (* The module's own bytes: the error record, then constant tables as the
     kernel asks for them, then the local region, then the host's buffers. *)
  let top = ref (align16 (Int64.of_int W.record_bytes)) in
  let mark_base =
    if count_marks then (
      let at = Layout.Address.of_int64 !top in
      top :=
        align16 (Int64.add !top (Int64.of_int (4 * List.length Loop_mark.all)));
      Some at)
    else None
  in
  let table_alloc ~bytes =
    let off = align8 !top in
    top := Int64.add off bytes;
    Layout.Address.of_int64 off
  in
  Err.map
    (fun (k : kernel) ->
      let imports, funcs, base =
        Loop_wasm_link.link ~callees:k.callees [ k.func ]
      in
      let local_base = Layout.Address.of_int64 (align16 !top) in
      let heap_base =
        Layout.Address.of_int64
          (align16
             (Int64.add (Layout.Address.to_int64 local_base) k.local_bytes))
      in
      let pages =
        max 1
          ((Int64.to_int (Layout.Address.to_int64 heap_base) + 65535) / 65536)
      in
      let m =
        {
          Wasm.Module.imports;
          funcs;
          globals = [];
          memory = Some { Wasm.Memory.min_pages = pages; max_pages = None };
          exports =
            [
              { Wasm.Export.name = "memory"; kind = Wasm.Export.Memory };
              {
                Wasm.Export.name = function_name;
                kind = Wasm.Export.Func (Loop_wasm_runtime.Func.to_int base);
              };
            ];
          data = k.data;
          customs = [ { Wasm.Custom.name = "abi"; payload = "loop-wasm/1" } ];
        }
      in
      let manifest =
        Loop_wasm_link.manifest ~numerics ~precisions:[ k.precision ]
          ~callees:k.callees m
      in
      {
        module_ =
          {
            m with
            Wasm.Module.customs =
              m.Wasm.Module.customs
              @ [ { Wasm.Custom.name = "manifest"; payload = manifest } ];
          };
        local_base;
        heap_base;
        mark_base;
        sites = k.sites;
        precision = k.precision;
      })
    (produce ~mark_base ~table_alloc)
