import { readFile, rename, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';

// PineTS strips venue prefixes for its built-in providers. Keep them for
// explicit local feeds so identical tickers on different venues stay distinct.
const original = new URL('./node_modules/pinets/dist/pinets.min.es.js', import.meta.url);
const source = new URL('./node_modules/pinets/dist/pinets.cyborg-0.13.1.1.es.js', import.meta.url);
const patches = [
    ['typeof t.tickerId=="string"&&t.tickerId.includes(":")?t.tickerId.split(":")[1]:t.tickerId', 't.tickerId'],
    ['typeof d=="string"&&d.includes(":")?d.split(":")[1]:d', 'd'],
    ['typeof c=="string"&&c.includes(":")?c.split(":")[1]:c', 'c'],
    ['if(Ir(v)===Ir(_))return', 'if(Ir(v)===Ir(_)&&u===t.tickerId)return'],
];

export async function loadRuntime() {
    let patched = await readFile(original, 'utf8');
    for (const [before, after] of patches) {
        if (patched.split(before).length !== 2) throw new Error('PineTS 0.13.1 bekleniyordu. npm ci ile kilitli sürümü kur.');
        patched = patched.replace(before, after);
    }
    const current = await readFile(source, 'utf8').catch(error => {
        if (error.code !== 'ENOENT') throw error;
    });
    if (current !== patched) {
        const temp = new URL(`${source.href}.${randomUUID()}.tmp`);
        await writeFile(temp, patched);
        await rename(temp, source);
    }
    return import(source.href);
}
