A module using every structural form, compiled and run under node.

  $ ./wasm_fixture_gen.exe fixture.wasm
  (module
    (import "m" "twice" (func 0 (param f64) (result f64)))
    (memory 1)
    (global 0 mut i32)
    (func 1 (param i32 i32) (result f64)
      (local f64 i32)
      block
        loop
          local.get 3
          local.get 1
          i32.ge_s
          br_if 1
          local.get 2
          local.get 0
          local.get 3
          i32.const 3
          i32.shl
          i32.add
          f64.load offset=0 align=8
          f64.add
          local.set 2
          local.get 3
          i32.const 1
          i32.add
          local.set 3
          br 0
        end
      end
      local.get 2
      call 0
    )
    (func 2 (param i64) (result i64)
      (local i64 i64)
      i64.const 1
      local.set 1
      block
        loop
          local.get 0
          i64.eqz
          br_if 1
          local.get 1
          local.get 0
          i64.mul
          local.set 1
          local.get 0
          i64.const 1
          i64.sub
          local.set 0
          br 0
        end
      end
      local.get 1
    )
    (func 3 (param i32 i32) (result i32)
      local.get 0
      if (result i32)
        local.get 1
        i32.load8_u offset=0 align=1
      else
        i32.const -1
      end
    )
    (func 4 (result i32)
      global.get 0
      i32.const 1
      i32.add
      global.set 0
      global.get 0
    )
    (func 5 (param i32 f64) (result f64)
      local.get 0
      local.get 1
      f32.demote_f64
      f32.store offset=0 align=4
      local.get 0
      f32.load offset=0 align=4
      f64.promote_f32
    )
    (func 6 (param i32) (result i32)
      local.get 0
      i32.const 7
      i32.const 4
      memory.fill
      local.get 0
      i32.const 8
      i32.add
      local.get 0
      i32.const 4
      memory.copy
      local.get 0
      i32.load offset=8 align=4
    )
    (export "memory" (memory 0))
    (export "sum_f64" (func 1))
    (export "fact" (func 2))
    (export "pick" (func 3))
    (export "bump" (func 4))
    (export "put_f32" (func 5))
    (export "counter" (global 0))
    (export "fill_copy" (func 6))
    (data (offset 16) 2 bytes)
  )
  $ node fixture.js fixture.wasm
  fixture ok
  negative checks ok
