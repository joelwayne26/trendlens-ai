import { classifyCategory } from '../src/lib/ai/category-rules';

console.log('TEST 1: "best food in uganda #uganda #food whatsapp us on 070735363" →',
  classifyCategory('best food in uganda #uganda #food whatsapp us on 070735363'));
console.log('TEST 2: "best food in uganda" →', classifyCategory('best food in uganda'));
console.log('TEST 3: "Try our luwombo with matooke" →', classifyCategory('Try our luwombo with matooke'));
console.log('TEST 4: "Birthday cake available! DM to order" →', classifyCategory('Birthday cake available! DM to order'));
