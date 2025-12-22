/* Linker script for Spartan Edge-7 Accelerator Board */

OUTPUT_FORMAT("elf32-littlearm", "elf32-littlearm", "elf32-littlearm")
OUTPUT_ARCH(arm)
ENTRY(_start)

MEMORY {
    DDR (rwx) : ORIGIN = 0x00100000, LENGTH = 0x1FF00000
    OCM (rwx) : ORIGIN = 0xFFFF0000, LENGTH = 0x00010000
    SRAM (rwx) : ORIGIN = 0x00000000, LENGTH = 0x00010000
}

SECTIONS {
    /* Code section */
    .text : {
        KEEP (*(.vectors))
        *(.text)
        *(.text.*)
        *(.gnu.linkonce.t.*)
        *(.plt)
        *(.gnu.warning)
        *(.glue_7)
        *(.glue_7t)
        *(.vfp11_veneer)
        *(.v4_bx)
    } > DDR

    /* Read-only data */
    .rodata : {
        __rodata_start = .;
        *(.rodata)
        *(.rodata.*)
        *(.gnu.linkonce.r.*)
        __rodata_end = .;
    } > DDR

    /* Initialized data */
    .data : {
        __data_start = .;
        *(.data)
        *(.data.*)
        *(.gnu.linkonce.d.*)
        __data_end = .;
    } > DDR

    /* Uninitialized data (BSS) */
    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        __bss_end = .;
    } > DDR

    /* Stack section */
    .stack (NOLOAD) : {
        . = ALIGN(8);
        __stack_start = .;
        . = . + 0x4000; /* 16KB stack */
        __stack_end = .;
    } > DDR

    /* Heap section */
    .heap (NOLOAD) : {
        . = ALIGN(8);
        __heap_start = .;
        . = . + 0x10000; /* 64KB heap */
        __heap_end = .;
    } > DDR

    /* Special sections for hardware accelerators */
    .hw_accel (NOLOAD) : {
        __hw_accel_start = .;
        KEEP(*(.hw_accel))
        __hw_accel_end = .;
    } > DDR

    /* DMA buffers */
    .dma_buffers (NOLOAD) : {
        __dma_buffers_start = .;
        KEEP(*(.dma_buffers))
        __dma_buffers_end = .;
    } > DDR

    /* Remove debugging information */
    /DISCARD/ : {
        *(.comment)
        *(.note)
        *(.note.*)
    }

    /* Symbol definitions for C code */
    __stack = __stack_end;
    __heap_size = __heap_end - __heap_start;
    __dma_buffer_size = __dma_buffers_end - __dma_buffers_start;
    
    /* End of memory */
    _end = .;
}
